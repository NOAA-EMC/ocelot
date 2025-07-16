#!/bin/bash

sudo apt update -y

# Install Python 3./10
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt install -y python3.10 python3.10-distutils python3.10-venv
