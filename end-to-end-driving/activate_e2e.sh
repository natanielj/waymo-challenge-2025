# set up e2e kernel

# initialize conda
eval "$(/workdir/crash-team/conda/bin/conda/ shell.bash hook)"

# activate env
conda activate e2e_vision

# install ipykernel
pip install --upgrade --quiet ipykernel

# make e2e_vision a jupyter kernel 
python -m ipykernel install --user --name=e2e_vision --display-name "Python (e2e_vision)"


echo "e2e vision kernel activated"
