import os
import cv2
import math
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLOE
from transformers import pipeline

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

IMAGE_PATH = "images/test-image-1.jpg"
DETECTION_OUTPUT = "results/detected.jpg"
GRAPH_OUTPUT = "results/scene_graph.png"
CONF_THRESHOLD = 0.50
PROXIMITY_THRESHOLD_PX = 250     # only compute relations between objects this close (pixels)
DEPTH_DIFF_THRESHOLD = 0.10      # relative depth difference needed to call "in front of/behind"
USE_DEPTH_RELATIONS = False      # see note above - leave off until scene has real depth variation

# Classes treated as "surfaces" rather than peer objects: they only ever appear as the
# target of an "X is on <surface>" relation, never compared left/right/front/behind
# against other objects (which was producing ~20 meaningless lines per run before).
SURFACE_CLASSES = {"desk", "table"}

CLASS_LIST = [
    "clipboard", "paper", "laptop", "monitor", "keyboard", "mouse",
    "cell phone", "tablet", "headphones", "cup", "pen", "book",
    "table", "chair", "desk", "mousepad", "camera", "toy",
    "wallet", "keys", "glasses", "perfume", "remote", "scissors",
]

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# STEP 1: OBJECT DETECTION (YOLOE)
# ----------------------------------------------------------------------
def run_detection(image_path):
    model = YOLOE("yoloe-26x-seg.pt")
    model.set_classes(CLASS_LIST)

    results = model.predict(source=image_path, device=DEVICE, conf=CONF_THRESHOLD)
    result = results[0]

    detections = []
    # running per-class counter so duplicate classes get distinct instance ids/labels,
    # e.g. two "mouse" detections become mouse_1 and mouse_2 internally
    class_counts = {}

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        name = model.names[class_id]

        class_counts[name] = class_counts.get(name, 0) + 1
        instance_id = f"{name}_{class_counts[name]}"

        detections.append({
            "id": instance_id,          # unique - use this internally for comparisons
            "name": name,               # display name - can repeat across detections
            "box": (x1, y1, x2, y2),
            "conf": confidence,
        })

        print(f"{name:15} confidence={confidence:.2f} bbox=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    # if any class had more than one detection, give it a numbered display label too,
    # so printed output can distinguish "mouse (1)" from "mouse (2)"
    for obj in detections:
        if class_counts[obj["name"]] > 1:
            obj["label"] = f"{obj['name']} ({obj['id'].split('_')[-1]})"
        else:
            obj["label"] = obj["name"]

    annotated = result.plot()
    cv2.imwrite(DETECTION_OUTPUT, annotated)
    print(f"\nSaved detection image: {DETECTION_OUTPUT}")

    return detections


# ----------------------------------------------------------------------
# STEP 2: DEPTH ESTIMATION (Depth Anything V2 - Base)
# ----------------------------------------------------------------------
def run_depth_estimation(image_path):
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Base-hf",
        device=DEVICE,
    )

    image = Image.open(image_path).convert("RGB")
    result = depth_pipe(image)

    depth_tensor = result["predicted_depth"]
    depth_resized = torch.nn.functional.interpolate(
        depth_tensor.unsqueeze(0).unsqueeze(0) if depth_tensor.dim() == 2 else depth_tensor.unsqueeze(0),
        size=(image.height, image.width),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    return depth_resized  # numpy array, shape (H, W), relative depth values


def get_object_depth(depth_array, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth_array.shape[1], x2), min(depth_array.shape[0], y2)
    region = depth_array[y1:y2, x1:x2]
    if region.size == 0:
        return None
    return float(np.median(region))


# ----------------------------------------------------------------------
# STEP 3: SPATIAL RELATIONS
# ----------------------------------------------------------------------
def get_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def get_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def distance(box_a, box_b):
    a_cx, a_cy = get_centroid(box_a)
    b_cx, b_cy = get_centroid(box_b)
    return math.hypot(a_cx - b_cx, a_cy - b_cy)


def is_on(box_a, box_b):
    """Is object A sitting ON object B? (B should be a larger surface)"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    if get_area(box_b) < get_area(box_a) * 1.5:
        return False

    a_cx, a_cy = get_centroid(box_a)
    if not (bx1 <= a_cx <= bx2):
        return False

    overlap_y = max(0, min(ay2, by2) - max(ay1, by1))
    a_height = ay2 - ay1
    return (overlap_y / a_height > 0.5) if a_height > 0 else False


def planar_relation(box_a, box_b):
    """left/right/above/below based on 2D centroid offset"""
    a_cx, a_cy = get_centroid(box_a)
    b_cx, b_cy = get_centroid(box_b)

    dx = a_cx - b_cx
    dy = a_cy - b_cy

    if abs(dx) > abs(dy):
        return "right of" if dx > 0 else "left of"
    else:
        return "below" if dy > 0 else "above"


def depth_relation(depth_a, depth_b):
    """in front of/behind based on relative depth values"""
    if depth_a is None or depth_b is None:
        return None
    diff = depth_a - depth_b
    if abs(diff) < DEPTH_DIFF_THRESHOLD:
        return None
    # NOTE: verify this sign on a known test case before trusting it -
    # Depth Anything's convention can vary; flip if front/behind come out backwards.
    return "behind" if diff > 0 else "in front of"


def build_scene_graph(detections, depth_array=None):
    if USE_DEPTH_RELATIONS and depth_array is not None:
        for obj in detections:
            obj["depth"] = get_object_depth(depth_array, obj["box"])

    triplets = []
    seen_pairs = set()  # avoid reporting both A-vs-B and B-vs-A

    for i, obj_a in enumerate(detections):
        for j, obj_b in enumerate(detections):
            if i == j:
                continue

            # surfaces (desk/table) are never a "left/right/on" PEER - they're only
            # ever the target of "X is on <surface>", handled by the surface check below
            if obj_a["name"] in SURFACE_CLASSES:
                continue

            pair_key = frozenset((obj_a["id"], obj_b["id"]))
            if pair_key in seen_pairs:
                continue

            if distance(obj_a["box"], obj_b["box"]) > PROXIMITY_THRESHOLD_PX:
                continue

            if obj_b["name"] in SURFACE_CLASSES:
                # only report this relation if A is actually ON the surface
                if is_on(obj_a["box"], obj_b["box"]):
                    triplets.append((obj_a["label"], "on", obj_b["label"]))
                    seen_pairs.add(pair_key)
                continue

            if is_on(obj_a["box"], obj_b["box"]):
                relation = "on"
            else:
                relation = planar_relation(obj_a["box"], obj_b["box"])

            triplets.append((obj_a["label"], relation, obj_b["label"]))
            seen_pairs.add(pair_key)

            if USE_DEPTH_RELATIONS:
                d_rel = depth_relation(obj_a.get("depth"), obj_b.get("depth"))
                if d_rel:
                    triplets.append((obj_a["label"], d_rel, obj_b["label"]))

    return triplets


# ----------------------------------------------------------------------
# STEP 4: SCENE GRAPH VISUALIZATION (node-edge relation graph)
# ----------------------------------------------------------------------
# Group relation types into a couple of colors so the legend stays simple
# instead of a different color per relation string.
RELATION_COLORS = {
    "on": "#0F6E56",  # containment/support - teal
    "left of": "#185FA5",  # planar - blue
    "right of": "#185FA5",
    "above": "#185FA5",
    "below": "#185FA5",
    "in front of": "#993C1D",  # depth - coral
    "behind": "#993C1D",
}
DEFAULT_EDGE_COLOR = "#5F5E5A"


def visualize_scene_graph(scene_graph, output_path=GRAPH_OUTPUT):
    """
    Build a directed node-edge graph from (subject, relation, object) triplets
    and save it as an image. Each object becomes a node; each relation becomes
    a labeled directed edge. Multiple edges between the same pair (e.g. both an
    'above' and an 'on' relation) are drawn as separate curved arcs so they
    don't overlap.
    """
    if not scene_graph:
        print("No relations to visualize - skipping scene graph image.")
        return

    G = nx.MultiDiGraph()
    for subj, rel, obj in scene_graph:
        G.add_edge(subj, obj, label=rel, color=RELATION_COLORS.get(rel, DEFAULT_EDGE_COLOR))

    pos = nx.spring_layout(G, seed=42, k=1.4 / math.sqrt(max(len(G.nodes), 1)))

    fig, ax = plt.subplots(figsize=(13, 10))

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color="#E6F1FB", edgecolors="#185FA5", linewidths=1.5, node_size=2600,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")

    # Draw each parallel edge with its own curvature so multi-edges don't overlap
    edge_groups = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        edge_groups.setdefault((u, v), []).append((k, data))

    for (u, v), edges in edge_groups.items():
        n = len(edges)
        for i, (k, data) in enumerate(edges):
            rad = 0.12 * (i - (n - 1) / 2) + (0.05 if u != v else 0.3)
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=[(u, v)],
                connectionstyle=f"arc3,rad={rad}",
                edge_color=data["color"], arrows=True, arrowsize=16,
                min_source_margin=28, min_target_margin=28, width=1.4,
            )
            (x1, y1), (x2, y2) = pos[u], pos[v]
            lx, ly = (x1 + x2) / 2 + rad * (y2 - y1) * 0.5, (y1 + y2) / 2 - rad * (x2 - x1) * 0.5
            ax.text(
                lx, ly, data["label"], fontsize=7, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1),
            )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scene graph visualization: {output_path}")

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Running object detection...\n")
    detections = run_detection(IMAGE_PATH)

    depth_array = None
    if USE_DEPTH_RELATIONS:
        print("\nRunning depth estimation...")
        depth_array = run_depth_estimation(IMAGE_PATH)

    print("\nBuilding scene graph...\n")
    scene_graph = build_scene_graph(detections, depth_array)

    print("Spatial relations:")
    for subj, rel, obj in scene_graph:
        print(f"  {subj} is {rel} {obj}")

    print("\nVisualizing scene graph...")
    visualize_scene_graph(scene_graph)