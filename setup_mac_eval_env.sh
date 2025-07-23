#!/bin/bash

# Exit on any error
set -e

# Create conda environment with Python 3.11
echo "Creating conda environment with Python 3.11..."
conda create -n pyserini python=3.11 -y

# Install core dependencies inside the conda environment
echo "Installing conda packages (wget, Java, Maven, faiss-gpu)..."
conda run -n pyserini conda install -c anaconda wget -y
conda run -n pyserini conda install -c conda-forge openjdk=21 maven -y
# Use faiss-cpu for CPU-only installations, or faiss-gpu for GPU support
conda run -n pyserini conda install -c pytorch faiss-cpu -y

# Upgrade pip and install Python packages
echo "Installing pip requirements..."
conda run -n pyserini pip install --upgrade pip

# Install torch explicitly first as flash_attn requires it
conda run -n pyserini pip install torch

# Make sure faiss-gpu is REMOVED from requirements.txt before this step
conda run -n pyserini pip install -r requirements.txt

# Clone and setup Anserini repository
echo "Cloning Anserini repository..."
git clone https://github.com/castorini/anserini.git --recurse-submodules
cd anserini
mvn clean package -Dmaven.test.skip=true
cd tools/eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make && cd ../../..
cd tools/eval/ndeval && make && cd ../../..
# This helps moves to the parent directory holding the anserini directory
cd ..



# Clone and setup Pyserini repository
echo "Cloning Pyserini repository..."
git clone https://github.com/castorini/pyserini.git --recurse-submodules
cd pyserini
cd tools/eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make && cd ../../..
cd tools/eval/ndeval && make && cd ../../..
pip install -e .
cd ..

# Move jar files from Anserini to Pyserini
echo "Moving jar files from Anserini to Pyserini..."
mv anserini/target/anserini-*-SNAPSHOT-fatjar.jar pyserini/pyserini/resources/jars/

echo "✅ JAR files moved successfully."

echo "✅ Environment setup complete."
echo "👉 Activate it with: conda activate pyserini"
