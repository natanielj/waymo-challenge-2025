
# create a conda environment
conda create --name "waymo-e2e" python=3.9
conda activate waymo-e2e

#upgrade pip 
pip install --upgrade pip

#install required packages
pip install -r requirements.txt

#register jupyter kernel
python -m ipykernel install --user --name waymo-e2e --display-name "waymo-e2e"