#!/bin/bash
# Sourced at the start of every job: detects what SLURM actually allocated and sizes the pipeline to it.
HERE="${BER_SLURM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$HERE/cluster.conf"
CODE="$(cd "$HERE/.." && pwd)"
SRC="$CODE/src"
[ -n "$MODULES" ] && eval "$MODULES"
eval "$ACTIVATE"

export BER_DATA_DIR BER_WORK_DIR BER_OUT_DIR HF_HOME BER_DENSE_MODEL BER_XENC_MODEL
mkdir -p "$BER_WORK_DIR" "$BER_OUT_DIR" "$HF_HOME"

# ---------------------------------------------------------------- CPUs / RAM of this allocation
NCPU="${SLURM_CPUS_PER_TASK:-$(nproc)}"
if [ -n "${SLURM_MEM_PER_NODE:-}" ] && [ "$SLURM_MEM_PER_NODE" -gt 0 ]; then
    MEM_GB=$(( SLURM_MEM_PER_NODE / 1024 ))
elif [ -n "${SLURM_MEM_PER_CPU:-}" ]; then
    MEM_GB=$(( SLURM_MEM_PER_CPU * NCPU / 1024 ))
else
    MEM_GB=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
fi

# ---------------------------------------------------------------- GPUs of this allocation
NGPU=0; GPU_MEM_GB=0; GPU_NAME="none"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]; then
        NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c . || true)
    else
        NGPU=$(nvidia-smi -L | grep -c "^GPU" || true)
    fi
    GPU_MEM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{print int($1 / 1024)}')
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
fi

# ---------------------------------------------------------------- CPU-side sizing
export BER_N_JOBS="$NCPU"
PROCS=$(( MEM_GB / 6 )); [ "$PROCS" -gt "$NCPU" ] && PROCS=$NCPU; [ "$PROCS" -gt 32 ] && PROCS=32; [ "$PROCS" -lt 2 ] && PROCS=2
export BER_N_PROCS="$PROCS"                       # RAM-heavy worker processes (blocking)
export OMP_NUM_THREADS="$NCPU" POLARS_MAX_THREADS="$NCPU" RAYON_NUM_THREADS="$NCPU"
if [ -z "$BER_N_TRAIN_S1" ]; then                 # more training S1 when the node can hold the features
    if   [ "$MEM_GB" -ge 400 ]; then BER_N_TRAIN_S1=500000
    elif [ "$MEM_GB" -ge 200 ]; then BER_N_TRAIN_S1=300000
    else                             BER_N_TRAIN_S1=150000; fi
fi
if [ -z "$BER_N_VAL_S1" ]; then
    if [ "$MEM_GB" -ge 200 ]; then BER_N_VAL_S1=80000; else BER_N_VAL_S1=50000; fi
fi
export BER_N_TRAIN_S1 BER_N_VAL_S1

# ---------------------------------------------------------------- GPU-side sizing (per GPU)
if   [ "$GPU_MEM_GB" -ge 70 ]; then            # H100 / A100 80 GB
    XB=512;  XIB=4096; XLR=4e-5; DB=1024; DEB=8192; TOPK=$(( 4 << 30 ))
elif [ "$GPU_MEM_GB" -ge 35 ]; then            # A100 40 GB, L40S, A6000
    XB=256;  XIB=2048; XLR=3e-5; DB=512;  DEB=4096; TOPK=$(( 2 << 30 ))
else                                           # 16-24 GB cards
    XB=128;  XIB=1024; XLR=2e-5; DB=256;  DEB=2048; TOPK=$(( 1 << 30 ))
fi
# LLM judge (full fine-tune of a ~1.5B decoder: fp32 master weights + AdamW ~ 24 GB before activations)
if   [ "$GPU_MEM_GB" -ge 70 ]; then LLM_BATCH=64; LLM_INF_BATCH=512; LLM_CKPT=0
elif [ "$GPU_MEM_GB" -ge 35 ]; then LLM_BATCH=32; LLM_INF_BATCH=256; LLM_CKPT=1
else                                LLM_BATCH=8;  LLM_INF_BATCH=128; LLM_CKPT=1; fi
case "$BER_LLM_MODEL" in *7B*|*8B*) LLM_BATCH=$(( LLM_BATCH / 4 )); LLM_CKPT=1 ;; esac
LLM_LR="${LLM_LR:-1e-5}"
export BER_LLM_MODEL LLM_BATCH LLM_INF_BATCH LLM_CKPT LLM_LR LLM_P_LO LLM_P_HI
export BER_XENC_BATCH="${BER_XENC_BATCH:-$XB}" BER_XENC_INF_BATCH="${BER_XENC_INF_BATCH:-$XIB}"
export BER_XENC_LR="${BER_XENC_LR:-$XLR}" BER_DENSE_BATCH="${BER_DENSE_BATCH:-$DB}"
export BER_DENSE_EMB_BATCH="${BER_DENSE_EMB_BATCH:-$DEB}" BER_TOPK_ELEMS="${BER_TOPK_ELEMS:-$TOPK}"
DP=$(( 1500000 * (NGPU > 0 ? NGPU : 1) )); [ "$DP" -gt 8000000 ] && DP=8000000
export BER_DENSE_PAIRS="${BER_DENSE_PAIRS:-$DP}"  # more contrastive pairs when more GPUs are available
LW=$(( NCPU / (NGPU > 0 ? NGPU : 1) - 1 )); [ "$LW" -gt 12 ] && LW=12; [ "$LW" -lt 2 ] && LW=2
export BER_LOADER_WORKERS="$LW"                   # tokeniser workers per GPU process

export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# offline if the models were pre-downloaded (compute nodes often have no internet)
if [ -d "$HF_HOME/hub/models--${BER_XENC_MODEL//\//--}" ]; then export HF_HUB_OFFLINE=1; fi

TORCHRUN="torchrun --standalone --nproc_per_node=${NGPU}"

echo "=== $(hostname) | job ${SLURM_JOB_ID:-local} | CPUs $NCPU | RAM ${MEM_GB} GB | GPUs $NGPU x $GPU_NAME (${GPU_MEM_GB} GB)"
echo "=== N_PROCS $BER_N_PROCS | train/val S1 $BER_N_TRAIN_S1/$BER_N_VAL_S1 | xenc batch $BER_XENC_BATCH (inf $BER_XENC_INF_BATCH, lr $BER_XENC_LR) | dense batch $BER_DENSE_BATCH, pairs $BER_DENSE_PAIRS | loader workers $BER_LOADER_WORKERS"
