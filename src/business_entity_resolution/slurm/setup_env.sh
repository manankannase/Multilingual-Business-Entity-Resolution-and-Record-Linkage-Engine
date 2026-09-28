#!/bin/bash
# Run ONCE on the login node (needs internet): Python env + model download into HF_HOME.
#   bash setup_env.sh            # CUDA 12.4 wheels (H100 ok)
#   CUDA_WHL=cu126 bash setup_env.sh
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster.conf"
[ -n "$MODULES" ] && eval "$MODULES"
CUDA_WHL="${CUDA_WHL:-cu124}"

python3 -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10 (set MODULES in cluster.conf)"'
[ -d "$VENV" ] || python3 -m venv "$VENV"
eval "$ACTIVATE"
pip install --upgrade pip
pip install torch --index-url "https://download.pytorch.org/whl/$CUDA_WHL"
pip install -r "$HERE/../requirements.txt" -r "$HERE/../requirements-gpu.txt"

export HF_HOME
mkdir -p "$HF_HOME"
python - <<EOF
from huggingface_hub import snapshot_download
for m in ["$BER_DENSE_MODEL", "$BER_XENC_MODEL", "$BER_LLM_MODEL"]:
    print("downloading", m, "->", snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]))
EOF
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__)"
echo "env ready: $VENV   models in $HF_HOME"
