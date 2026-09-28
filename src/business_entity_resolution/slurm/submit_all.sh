#!/bin/bash
# Submit the whole pipeline as a SLURM dependency chain, sizing every job from `sinfo`:
#
#   prep ──► cand ──────────┐           ┌─► xenc (GPU) ─┐
#     └────► dense (GPU) ───┴─► feat ───┤               ├─► stage2
#                                       └─► llm  (GPU) ─┘
#
# Usage:  bash submit_all.sh                 # stages already finished (.done_<stage>) are skipped
#         FROM=xenc bash submit_all.sh       # force re-run from a stage onwards
#         SKIP_DENSE=1 / SKIP_XENC=1 / SKIP_LLM=1   # leave out a GPU stage
#         DRY=1 bash submit_all.sh           # print the sbatch commands only
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster.conf"
LOGS="$BER_WORK_DIR/logs"; mkdir -p "$LOGS"

# ---------------------------------------------------------------- detect partition & node shape
if [ -z "$GPU_PARTITION" ]; then
    GPU_PARTITION=$(sinfo -h -o "%P %G" | grep -i "h100" | head -1 | awk '{print $1}' | tr -d '*' || true)
    [ -n "$GPU_PARTITION" ] || { echo "no partition with H100 GRES found; set GPU_PARTITION in cluster.conf"; exit 1; }
fi
CPU_PARTITION="${CPU_PARTITION:-$GPU_PARTITION}"

node_shape() {   # partition -> "cpus mem_mb gres" of its largest node
    sinfo -h -p "$1" -o "%c %m %G" | sort -k1,1n -k2,2n | tail -1
}
read -r G_CPUS G_MEM G_GRES <<< "$(node_shape "$GPU_PARTITION")"
read -r C_CPUS C_MEM _ <<< "$(node_shape "$CPU_PARTITION")"
G_CPUS=${G_CPUS%%[!0-9]*}; G_MEM=${G_MEM%%[!0-9]*}; C_CPUS=${C_CPUS%%[!0-9]*}; C_MEM=${C_MEM%%[!0-9]*}
# gres looks like gpu:h100:8(S:0-1) or gpu:8
GPU_TYPE=$(echo "$G_GRES" | grep -oiE "gpu:[a-z0-9_]*h100[a-z0-9_]*" | head -1 | cut -d: -f2 || true)
NODE_GPUS=$(echo "$G_GRES" | grep -oE "gpu(:[A-Za-z0-9_]+)?:[0-9]+" | head -1 | awk -F: '{print $NF}' || true)
NODE_GPUS=${NODE_GPUS:-1}; G_CPUS=${G_CPUS:-32}; G_MEM=${G_MEM:-256000}; C_CPUS=${C_CPUS:-32}; C_MEM=${C_MEM:-256000}

if [ -z "$GPUS" ]; then GPUS=$(( NODE_GPUS < 4 ? NODE_GPUS : 4 )); fi
GRES="gpu:${GPU_TYPE:+$GPU_TYPE:}$GPUS"
GPU_JOB_CPUS=$(( G_CPUS * GPUS / NODE_GPUS ))
GPU_JOB_MEM=$(( G_MEM * GPUS / NODE_GPUS * 95 / 100 ))
if [ -z "$CPU_CPUS" ]; then CPU_CPUS=$(( C_CPUS < 64 ? C_CPUS : 64 )); fi
CPU_JOB_MEM=$(( C_MEM * CPU_CPUS / C_CPUS ))
MIN_MEM=$(( C_MEM * 90 / 100 < 128000 ? C_MEM * 90 / 100 : 128000 ))  # CPU stages want >= 128 GB
[ "$CPU_JOB_MEM" -lt "$MIN_MEM" ] && CPU_JOB_MEM=$MIN_MEM
CPU_GRES=""; [ "$CPU_STAGE_GPUS" -gt 0 ] 2>/dev/null && CPU_GRES="--gres=gpu:${GPU_TYPE:+$GPU_TYPE:}$CPU_STAGE_GPUS"

echo "GPU jobs : partition $GPU_PARTITION, --gres=$GRES, $GPU_JOB_CPUS CPUs, $(( GPU_JOB_MEM / 1024 )) GB (node: $NODE_GPUS GPUs, $G_CPUS CPUs, $(( G_MEM / 1024 )) GB)"
echo "CPU jobs : partition $CPU_PARTITION, $CPU_CPUS CPUs, $(( CPU_JOB_MEM / 1024 )) GB ${CPU_GRES}"

# ---------------------------------------------------------------- submit
COMMON=(--nodes=1 --ntasks=1 --export=ALL,BER_SLURM_DIR="$HERE")
[ -n "$ACCOUNT" ] && COMMON+=(--account="$ACCOUNT")
[ -n "$QOS" ] && COMMON+=(--qos="$QOS")
ORDER=(prep cand dense feat xenc llm stage2)
forced=0
declare -A JOB

submit() {   # stage kind time deps...
    local st=$1 kind=$2 tlim=$3; shift 3
    [ "$st" = "$FROM" ] && forced=1
    if [ $forced -eq 0 ] && [ -f "$BER_WORK_DIR/.done_$st" ]; then echo "skip $st (done)"; return; fi
    [ $forced -eq 1 ] && rm -f "$BER_WORK_DIR/.done_$st"
    local dep=() ids=()
    for d in "$@"; do [ -n "${JOB[$d]:-}" ] && ids+=("${JOB[$d]}"); done
    [ ${#ids[@]} -gt 0 ] && dep=(--dependency=afterok:$(IFS=:; echo "${ids[*]}") --kill-on-invalid-dep=yes)
    local res
    if [ "$kind" = gpu ]; then
        res=(--partition="$GPU_PARTITION" --gres="$GRES" --cpus-per-task="$GPU_JOB_CPUS" --mem="${GPU_JOB_MEM}M")
    else
        res=(--partition="$CPU_PARTITION" --cpus-per-task="$CPU_CPUS" --mem="${CPU_JOB_MEM}M" $CPU_GRES)
    fi
    local cmd=(sbatch --parsable --job-name="ber_$st" --time="$tlim" --output="$LOGS/%x_%j.out" --error="$LOGS/%x_%j.err"
               "${COMMON[@]}" "${res[@]}" "${dep[@]}" "$HERE/run_stage.sh" "$st")
    if [ -n "$DRY" ]; then echo "${cmd[@]}"; JOB[$st]="DRY_$st"; return; fi
    JOB[$st]=$("${cmd[@]}")
    echo "submitted $st -> job ${JOB[$st]}"
}

[ -n "$FROM" ] && [[ ! " ${ORDER[*]} " =~ " $FROM " ]] && { echo "FROM must be one of ${ORDER[*]}"; exit 2; }
submit prep   cpu "$TIME_PREP"
submit cand   cpu "$TIME_CAND"   prep
[ -z "$SKIP_DENSE" ] && submit dense gpu "$TIME_DENSE" prep
submit feat   cpu "$TIME_FEAT"   cand dense
[ -z "$SKIP_XENC" ] && submit xenc gpu "$TIME_XENC" feat
[ -z "$SKIP_LLM" ] && submit llm gpu "$TIME_LLM" feat
submit stage2 cpu "$TIME_STAGE2" feat xenc llm
echo "logs: $LOGS  (<stage>_<jobid>.out = progress, .err = warnings/errors)   status: bash $HERE/status.sh"
