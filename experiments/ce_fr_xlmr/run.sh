#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
python -c 'import sys; assert sys.version_info >= (3,11), "Use Python 3.11 or 3.12"'
exec python -u train_ce.py "$@"
