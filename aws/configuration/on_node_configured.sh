#!/usr/bin/env bash

python3.10 -m venv venv
source venv/bin/activate
echo "source ~/venv/bin/activate" >> ~/.bashrc

python3.10 -m pip install --upgrade pip

python3.10 -m pip install numpy==1.26.4
python3.10 -m pip install pandas==2.2.2
python3.10 -m pip install torch==2.5.1
python3.10 -m pip install torch_scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
python3.10 -m pip install torch-geometric==2.6.1
python3.10 -m pip install lightning==2.5.1
python3.10 -m pip install scikit-learn==1.6.1
python3.10 -m pip install matplotlib==3.9.4
python3.10 -m pip install psutil==5.9.8
python3.10 -m pip install trimesh==4.6.10
python3.10 -m pip install zarr==2.18.0

git clone https://github.com/NOAA-EMC/ocelot.git
git checkout main
