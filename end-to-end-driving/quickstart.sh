# Waymo Environment Setup Script

set -e  # stop if any command fails

echo "=== Linking Waymo dataset to ~/work/waymo_e2e ==="
mkdir -p ~/work
ln -sf /data/waymo_e2e ~/work/waymo_e2e

echo "=== Upgrading pip, setuptools, and wheel ==="
python3 -m pip install --upgrade pip setuptools wheel

# echo "=== Installing TensorFlow (system level for safety) ==="
# python3 -m pip install tensorflow==2.12.0

echo "=== Creating Conda environment: waymo ==="
conda create -n waymo python=3.10 -y

echo "=== Activating Conda environment ==="

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate waymo

echo "=== Installing Python packages ==="
pip install --upgrade pip setuptools wheel
pip install tensorflow==2.12.0
pip install waymo-open-dataset-tf-2-12-0==1.6.5
pip install ipykernel opencv-python-headless matplotlib numpy

echo "=== Registering Jupyter kernel ==="
python -m ipykernel install --user --name waymo --display-name "Python (waymo)"

echo "=== Waymo environment setup complete ==="
echo "To use it in Jupyter, select kernel: Python (waymo)"
