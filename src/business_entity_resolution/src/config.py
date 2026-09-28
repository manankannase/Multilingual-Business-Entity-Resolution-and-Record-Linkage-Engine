"""Paths and global settings. Override locations with environment variables."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get(
    "BER_DATA_DIR",
    "D:/mlchallenge/6ab10eb3b23ba_student_resource/student_resource/dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", "D:/mlchallenge/work"))
OUT_DIR = Path(os.environ.get("BER_OUT_DIR", "D:/mlchallenge/submission/output"))


def _cpus() -> int:
    """CPUs this process may use: SLURM allocation > affinity mask > machine count (os.cpu_count() reports the
    whole node on a cluster, which oversubscribes a partial allocation)."""
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 4


N_JOBS = int(os.environ.get("BER_N_JOBS", _cpus()))                        # threads (matmul, rapidfuzz)
N_PROCS = int(os.environ.get("BER_N_PROCS", min(6, _cpus())))              # processes (RAM-heavy)
SEED = 42

# train/val sample sizes (whole states); raise on a big-memory node for a stronger stage-1 model
N_TRAIN_S1 = int(os.environ.get("BER_N_TRAIN_S1", 150_000))
N_VAL_S1 = int(os.environ.get("BER_N_VAL_S1", 50_000))

for _d in (WORK_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
