# CPEN355-Final-Project

## ⚙️ Setup (uv)
### 0. uv install if you don't have uv

### For Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

### For MacOS
curl -Ls https://astral.sh/uv/install.sh | sh

### 1. Create virtual environment
uv venv

### 2. Activate environment

macOS / Linux:
source .venv/bin/activate

Windows:
.venv\Scripts\activate

### 2.1 Download CUDA for Windows NVIDIA Users
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

### 3. Install dependencies
uv pip install -r requirements.txt

### 4. Run the pipeline
python entry.py

### Verify environment (important)
which python

Expected:
.../.venv/bin/python

