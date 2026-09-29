# rc-car-with-scene-understanding-and-route-calculation
This repository is for code of an rc car with a camera mount which generates a route to a target while evading objects.

## Getting Started

### Prerequisites

- [Python 3.9+](https://www.python.org/downloads/) (check with `python --version`)

### 1. Clone the repository

```bash
git clone https://github.com/m-saifullah3/rc-car-with-scene-understanding-and-route-calculation.git
cd https://github.com/m-saifullah3/rc-car-with-scene-understanding-and-route-calculation.git
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv

# macOS / Linux
python3 -m venv venv
```

### 3. Activate the virtual environment

```bash
# Windows (Command Prompt)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

When it's active, `(venv)` appears at the start of your terminal prompt.

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure the Python interpreter

**VS Code**
1. Open the project folder (`File > Open Folder`).
2. Press `Ctrl+Shift+P` (`Cmd+Shift+P` on macOS) and choose **Python: Select Interpreter**.
3. Pick the one located in the project's `venv` folder:
   - Windows: `.\venv\Scripts\python.exe`
   - macOS / Linux: `./venv/bin/python`

**PyCharm**
1. Go to `File > Settings > Project > Python Interpreter` (`PyCharm > Settings` on macOS).
2. Click **Add Interpreter > Add Local Interpreter > Existing**.
3. Select the `python` executable inside the `venv` folder and click **OK**.

### 6. Run the project

```bash
python main.py
```

### 7. Deactivate the environment when finished

```bash
deactivate
```
