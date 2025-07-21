#!/bin/bash

# Ensure the script fails on any error and variables are defined
set -euo pipefail

# Install Python 3.10 non-interactively
export DEBIAN_FRONTEND=noninteractive
#sudo apt-get update -y
#sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.10 python3.10-distutils python3.10-venv
