#!/bin/bash

# Ensure the script fails on any error and variables are defined
set -euo pipefail

# Install Python 3.10 non-interactively
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.10 python3.10-distutils python3.10-venv
