#!/usr/bin/env bash
#
# One-shot script to create a conda env + dev builds of Anserini + Pyserini
#

# Accept ToS for Anaconda channels
# Remove these lines (not supported on macOS)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

set -euo pipefail

ENV_NAME=pyserini-dev
PY_VER=3.11
ROOT=$(pwd)                     # assume you run this from repo root
LIBS=$ROOT/libs

echo "📦  Creating conda env '${ENV_NAME}' (Python ${PY_VER}) ..."
conda create -y -n $ENV_NAME python=$PY_VER
conda run -n $ENV_NAME conda install -y -c conda-forge openjdk=21 maven wget
conda run -n $ENV_NAME conda install -y -c pytorch faiss-cpu   # or faiss-gpu
conda run -n $ENV_NAME pip install --upgrade pip torch

# ------------------------------------------------------------------
# Clone / build Anserini (Java)
# ------------------------------------------------------------------
mkdir -p "$LIBS"
if [ ! -d "$LIBS/anserini" ]; then
  git clone https://github.com/castorini/anserini.git --recurse-submodules "$LIBS/anserini"
fi

echo "🔨  Building Anserini fat JAR ..."
pushd "$LIBS/anserini" >/dev/null
conda run -n $ENV_NAME mvn clean package -Dmaven.test.skip=true
(cd tools/eval && tar --no-same-owner -xvzf trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make)
(cd tools/eval/ndeval && make)
popd >/dev/null

# ------------------------------------------------------------------
# Clone / install Pyserini (editable)
# ------------------------------------------------------------------
if [ ! -d "$LIBS/pyserini" ]; then
  git clone https://github.com/castorini/pyserini.git --recurse-submodules "$LIBS/pyserini"
fi

echo "🔧  Compiling trec_eval / ndeval helpers ..."
pushd "$LIBS/pyserini" >/dev/null
(cd tools/eval && tar --no-same-owner -xvzf trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make)
(cd tools/eval/ndeval && make)
conda run -n $ENV_NAME pip install -e .
popd >/dev/null

# ------------------------------------------------------------------
# Copy Anserini fat-jar into Pyserini’s resources
# ------------------------------------------------------------------
echo "📁  Copying Anserini fat JAR into Pyserini ..."
cp "$LIBS/anserini"/target/anserini-*-SNAPSHOT-fatjar.jar \
   "$LIBS/pyserini/pyserini/resources/jars/"

# Make sure faiss-gpu is REMOVED from requirements.txt before this step
conda run -n $ENV_NAME pip install -r requirements.txt

# Install vllm for query generation
# echo "Installing vllm..."
# conda run -n $ENV_NAME pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128

echo -e "\n✅  Dev environment ready:"
echo "   • conda activate ${ENV_NAME}"
echo "   • run your scripts: python scripts/.../my_script.py"
