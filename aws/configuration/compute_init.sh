
sudo apt install python3 -y
sudo apt install python3-pip -y
sudo apt install python3-virtualenv -y
sudo apt install git -y

python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

python -m pip install torch==2.5.1
python -m pip install torch-geometric==2.6.1
python -m pip install torch_scatter==2.1.2
python -m pip install lightning==2.5.1
python -m pip install numpy==1.26.4
python -m pip install pandas==2.2.2
python -m pip install scikit-learn==1.6.1
python -m pip install scipy==1.15.2
python -m pip install matplotlib==3.10.3
python -m pip install psutil==5.9.8
python -m pip install trimesh==4.6.10
python -m pip install networkx==3.2.1
python -m pip install zarr==2.18.0
python -m pip install tqdm==4.67.1
python -m pip install pyyaml==6.0.2

git clone https://github.com/NOAA-EMC/ocelot.git



