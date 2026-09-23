from ultralytics import YOLO, YOLOE
import cv2

# Load a pretrained YOLOE segmentation model
model = YOLOE("yoloe-26x-seg.pt")

model.set_classes([
    "laptop",
    "computer monitor",
    "keyboard",
    "mouse",
    "cell phone",
    "tablet",
    "headphones",
    "cup",
    "pen",
    "book",
    "table",
    "chair",
    "desk"
])

# Input image
image_path = "images/test-image-1.jpg"

# Run detection on the GPU
results = model.predict(
    source=image_path,
    device=0,
    conf=0.20
)

result = results[0]

# Print detected objects
for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    name = model.names[class_id]

    print(
        f"{name:15} "
        f"confidence={confidence:.2f} "
        f"bbox=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
    )

# Draw detections on the image
annotated = result.plot()

cv2.imwrite("results/detected.jpg", annotated)

print("\nSaved: results/detected.jpg")