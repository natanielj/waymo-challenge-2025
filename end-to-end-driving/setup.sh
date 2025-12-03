#!/usr/bin/env bash
set -euo pipefail

# ---------- Config (override via env or args) ----------
ENV_NAME="${1:-waymo-env}"              # usage: bash setup.sh [env_name]
PY_VER="${PY_VER:-3.10}"
TF_VARIANT="${TF_VARIANT:-gpu}"        # gpu | cpu
KERNEL_DISPLAY="${KERNEL_DISPLAY:-Python ${PY_VER} (Waymo TF2.13)}"
REQ_FILE="${REQ_FILE:-requirements-waymo.txt}"

# ---------- Conda bootstrap ----------
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Please install Miniconda/Anaconda and retry." >&2
  exit 1
fi

# Enable 'conda activate' in non-interactive shells
# (use either of the two lines below; first one is most portable)
eval "$(conda shell.bash hook)"
# source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------- Create & activate env ----------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "ℹ️  Conda env '${ENV_NAME}' already exists. Reusing it."
else
  echo "🛠️  Creating conda env '${ENV_NAME}' (python=${PY_VER})..."
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

echo "➡️  Activating '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

# ---------- Pip + wheel tooling ----------
python -m pip install --upgrade pip wheel

# ---------- Write exact, known-good requirements ----------
# Waymo 1.6.7 pins several deps (TF 2.13, NumPy 1.23.5, etc.)
# We also keep seaborn at 0.11.2 because mpl 3.6.1 is required by Waymo and seaborn 0.12+ forbids it.
echo "📄 Writing ${REQ_FILE}..."
cat > "${REQ_FILE}" <<'EOF'
# Core pins for Waymo 1.6.7 (tf-2-12-0 wheel)
tensorflow==2.13
waymo-open-dataset-tf-2-12-0==1.6.7

# Waymo-required pins
numpy==1.23.5
absl-py==1.4.0
google-auth==2.16.2
protobuf==4.23.4
matplotlib==3.6.1
dask[dataframe]==2023.3.1
distributed==2023.3.1
setuptools==67.6.0

# Compatible extras
seaborn==0.11.2
opencv-python-headless==4.7.0.72
typing-extensions>=4.10,<5

# Jupyter kernel support
ipykernel
EOF

# If CPU-only requested, swap tensorflow package name before install
if [[ "${TF_VARIANT}" == "cpu" ]]; then
  echo "⚙️  Using CPU-only TensorFlow."
  # Replace 'tensorflow==2.13' with 'tensorflow-cpu==2.13'
  sed -i.bak 's/^tensorflow==2\.13$/tensorflow-cpu==2.13/' "${REQ_FILE}"
else
  echo "⚙️  Using GPU-enabled TensorFlow (pip wheel)."
fi

# ---------- Install requirements ----------
echo "⬇️  Installing pinned requirements..."
python -m pip install --no-cache-dir --force-reinstall -r "${REQ_FILE}"

# ---------- Register Jupyter kernel ----------
echo "🧩 Registering Jupyter kernel '${ENV_NAME}' → '${KERNEL_DISPLAY}'..."
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "${KERNEL_DISPLAY}"

# ---------- Sanity check ----------
python - <<'PY'
import sys
pkgs = [
  ("tensorflow", "import tensorflow as tf; print('TF', tf.__version__)"),
  ("waymo_open_dataset", "from waymo_open_dataset import dataset_pb2; print('Waymo OK')"),
  ("numpy", "import numpy as np; print('NumPy', np.__version__)"),
  ("cv2", "import cv2; print('OpenCV', cv2.__version__)"),
  ("matplotlib", "import matplotlib; print('mpl', matplotlib.__version__)"),
  ("seaborn", "import seaborn; print('seaborn', seaborn.__version__)"),
  ("dask", "import dask; print('dask', dask.__version__)"),
  ("distributed", "import distributed; print('distributed', distributed.__version__)"),
]
print(f"Python: {sys.version.split()[0]}  @ {sys.executable}")
for name, code in pkgs:
    try:
        exec(code, {})
    except Exception as e:
        print(f"{name}: FAILED -> {e.__class__.__name__}: {e}")
PY

echo ""
echo "✅ Done."
echo "   • Start Jupyter, then pick kernel: '${KERNEL_DISPLAY}'"
echo "   • To reinstall TF for CPU-only later:  TF_VARIANT=cpu bash setup.sh ${ENV_NAME}"
