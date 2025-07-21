#!/bin/bash

set -euo pipefail

python3.10 -m pip install --no-input --upgrade pip

python3.10 -m pip install --no-input numpy==1.26.4
python3.10 -m pip install --no-input pandas==2.2.2
python3.10 -m pip install --no-input torch==2.5.1
python3.10 -m pip install --no-input torch_scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
python3.10 -m pip install --no-input torch-geometric==2.6.1
python3.10 -m pip install --no-input lightning==2.5.1
python3.10 -m pip install --no-input scikit-learn==1.6.1
python3.10 -m pip install --no-input matplotlib==3.9.4
python3.10 -m pip install --no-input psutil==5.9.8
python3.10 -m pip install --no-input trimesh==4.6.10
python3.10 -m pip install --no-input zarr==2.18.0

