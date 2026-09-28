"""Shared GPU plumbing for gpu_dense.py (bi-encoder blocking leg) and gpu_xenc.py (cross-encoder re-ranker).

Throughput choices (keep the GPU busy, never the CPU):
  * torchrun / DDP: one process per GPU, work split by batch (b % world == rank).
  * texts live in one packed byte buffer (TextTable) -> DataLoader workers fork without copying millions of
    Python strings; tokenisation happens inside the workers, overlapped with GPU compute.
  * length-bucketed batches -> almost no padding; pad_to_multiple_of=8 for tensor cores.
  * bf16 autocast (fp16 + GradScaler on GPUs without bf16), TF32 matmuls, SDPA attention, fused AdamW,
    pinned memory + non_blocking copies.
"""
import os
import time

import numpy as np
import polars as pl
import pyarrow as pa
import torch
import torch.distributed as dist

from config import N_JOBS, SEED, WORK_DIR

MODEL_DIR = WORK_DIR / "models"
TEXT_COLS = ["entity_id", "name_clean", "addr_clean", "state"]


# ------------------------------------------------------------------------------------------ distributed
class Dist:
    def __init__(self):
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.local = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local)
            self.device = torch.device("cuda", self.local)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
        if self.world > 1:
            dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo")
        self.main = self.rank == 0

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def log(self, *a):
        if self.main:
            print(time.strftime("%H:%M:%S"), *a, flush=True)

    def close(self):
        if self.world > 1:
            dist.destroy_process_group()


def amp_dtype(device) -> torch.dtype:
    if device.type != "cuda":
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_tokenizer(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, use_fast=True)


def load_model(cls, name: str, **kw):
    """HF model with fused SDPA attention when the architecture supports it."""
    try:
        return cls.from_pretrained(name, attn_implementation="sdpa", **kw)
    except (ValueError, TypeError, ImportError):
        return cls.from_pretrained(name, **kw)


def wrap_ddp(model, d: "Dist"):
    if d.world == 1:
        return model
    ids = [d.local] if d.device.type == "cuda" else None
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=ids)


def maybe_compile(model):
    if os.environ.get("BER_COMPILE", "0") == "1":
        return torch.compile(model, dynamic=True)
    return model


# ------------------------------------------------------------------------------------------ texts
class TextTable:
    """Millions of strings as one uint8 buffer + int64 offsets (fork-safe, no per-string Python objects)."""

    def __init__(self, s: pl.Series):
        arr = s.fill_null("").cast(pl.Utf8).rechunk().to_arrow()
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        arr = arr.cast(pa.large_string())
        _, off, data = arr.buffers()
        o = np.frombuffer(off, np.int64)[arr.offset:arr.offset + len(arr) + 1]
        base = int(o[0])
        self.off = (o - base).astype(np.int64)
        self.buf = (np.frombuffer(data, np.uint8)[base:base + int(self.off[-1])].copy()
                    if data is not None else np.zeros(0, np.uint8))

    def __len__(self):
        return len(self.off) - 1

    def get(self, idx) -> list:
        o, b = self.off, self.buf
        return [b[o[i]:o[i + 1]].tobytes().decode("utf-8", "replace") for i in idx]

    def lengths(self) -> np.ndarray:
        return np.diff(self.off)


def text_expr() -> pl.Expr:
    """What the transformers read for one record: normalised (Indic already transliterated by the learned
    dictionary, legal forms canonicalised) name | address | state code."""
    return pl.concat_str([pl.col("name_clean"), pl.col("addr_clean"), pl.col("state")],
                         separator=" | ").alias("text")


def records(split: str, srcs, ids: pl.Series = None, country: str = None, extra=()) -> pl.DataFrame:
    """entity_id, text (+ extra columns) for sources `srcs` (file order kept), lazily filtered to `ids` /
    `country` when given."""
    cols = list(dict.fromkeys(TEXT_COLS + ["country"] + list(extra)))
    frames = []
    for s in srcs:
        lf = pl.scan_parquet(WORK_DIR / "norm" / f"{split}_source{s}.parquet").select(cols)
        if country is not None:
            lf = lf.filter(pl.col("country") == country)
        if ids is not None:
            lf = lf.filter(pl.col("entity_id").is_in(ids.implode()))
        frames.append(lf.select("entity_id", text_expr(), *extra).collect())
    return pl.concat(frames)


# ------------------------------------------------------------------------------------------ batching
def length_batches(lengths: np.ndarray, batch: int, shuffle: bool, seed: int = SEED, bucket: int = 64):
    """Index batches of similar length. shuffle=True: random mega-buckets (bucket*batch rows) sorted inside,
    batch order shuffled (training). shuffle=False: global sort (inference)."""
    n = len(lengths)
    if shuffle:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        out = []
        mb = batch * bucket
        for s in range(0, n, mb):
            chunk = perm[s:s + mb]
            chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
            out.extend(chunk[i:i + batch] for i in range(0, len(chunk), batch))
        out = [b for b in out if len(b) == batch]            # equal shapes for DDP / all_gather
        rng.shuffle(out)
        return out
    order = np.argsort(lengths, kind="stable")[::-1]         # longest first: OOM shows up immediately
    return [order[i:i + batch] for i in range(0, n, batch)]


def rank_share(batches: list, d: Dist, equal: bool) -> list:
    mine = batches[d.rank::d.world]
    if equal:                                                # DDP needs the same number of steps per rank
        mine = mine[:len(batches) // d.world]
    return mine


class _BatchDS(torch.utils.data.Dataset):
    """Each item is a whole batch (list of row indices) -> tokenised inside the DataLoader worker."""

    def __init__(self, batches, make):
        self.batches, self.make = batches, make

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return self.make(self.batches[i])


def _worker_init(_):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)


def loader(batches, make, workers: int = None):
    workers = workers if workers is not None else int(os.environ.get(
        "BER_LOADER_WORKERS", max(2, min(12, N_JOBS // max(1, torch.cuda.device_count())))))
    return torch.utils.data.DataLoader(
        _BatchDS(batches, make), batch_size=None, shuffle=False, num_workers=workers,
        pin_memory=torch.cuda.is_available(), worker_init_fn=_worker_init,
        prefetch_factor=4 if workers > 0 else None, persistent_workers=False)


def to_dev(batch: dict, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def lr_schedule(opt, total: int, warmup: float = 0.05):
    """Linear warm-up then linear decay to 0."""
    w = max(1, int(total * warmup))

    def f(s):
        return (s + 1) / w if s < w else max(0.0, (total - s) / max(1, total - w))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def adamw(model, lr: float, wd: float = 0.01):
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    try:
        return torch.optim.AdamW(groups, lr=lr, fused=torch.cuda.is_available())
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(groups, lr=lr)


def gpu_mem() -> str:
    if not torch.cuda.is_available():
        return ""
    return f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
