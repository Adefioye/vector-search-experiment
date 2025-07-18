#!/bin/bash

# Exit on any error
set -e

# Create conda environment
echo "Creating conda environment..."
conda create -n pyserini python=3.11 -y

# Activate environment
# For non-interactive scripts, we must use `conda run`
echo "Installing dependencies in pyserini environment..."
conda run -n pyserini conda install -c anaconda wget -y
conda run -n pyserini conda install -c conda-forge openjdk=21 maven -y

# Install pip dependencies inside the conda environment
echo "Installing Python requirements..."
conda run -n pyserini pip install --upgrade pip
conda run -n pyserini pip install -r requirements.txt

echo "✅ Environment setup complete. Use 'conda activate pyserini' to start working."
