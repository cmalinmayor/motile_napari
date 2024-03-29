# Setting up the Environment

## Create a conda env with napari graph support
```sh
conda create -n napari_graph python=3.10 pyqt=5.15.7 # 3.12 and 3.11 not yet supported
conda activate napari_graph
pip install 'git+https://github.com/JoOkuma/napari@napari-graph-2023#egg=napari'
pip install 'git+https://github.com/napari/napari-graph'
```

## Install Motile
```sh
conda install -c conda-forge -c funkelab -c gurobi ilpy
pip install git+https://github.com/funkelab/motile.git@1722dabd6e314d349ff50d46fd61abbdd9c17845
```

## Install Traccuracy
```sh
pip install 'git+https://github.com/Janelia-Trackathon-2023/traccuracy'
```

## Misc Other Packages
```sh
pip install jupytext zarr pre-commit
```

## To run script as notebook
- run jupyter lab (make sure jupytext extension is enabled)
- Right click on python script `run_motile.py`
- Open with -> jupytext notebook
Note: we are using the format "percent script"
