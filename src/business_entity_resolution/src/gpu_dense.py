"""GPU dense blocking leg: fine-tuned multilingual bi-encoder + exact top-k search on the GPU, per country.

  torchrun --nproc_per_node=G gpu_dense.py train           # contrastive fine-tune (in-batch negatives)
  torchrun --nproc_per_node=G gpu_dense.py search train    # dense candidates for the train/val S1 subsets
  torchrun --nproc_per_node=G gpu_dense.py search test
  python pipeline.py merge_dense train|test                # adds sim/rk_{f,r}_dense to the candidate shards

The encoder is trained only on ground-truth pairs of train S1 that are NOT in the LightGBM train/val subsets:
the dense similarity becomes a LightGBM feature, and pairs the encoder has seen would look more similar than
test pairs. Forward leg: every query S1 -> top-k pool records of its country. Reverse leg: every pool record ->
top-k S1 of its country (all S1 compete, like the TF-IDF reverse leg).
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F

from config import SEED
from data import load_ground_truth
from gpu_common import (MODEL_DIR, Dist, TextTable, adamw, amp_dtype, gpu_mem, length_batches, load_model,
                        load_tokenizer, loader, lr_schedule, maybe_compile, rank_share, records, to_dev,
                        wrap_ddp)
from pipeline import DENSE, countries, s1_subset

BASE = os.environ.get("BER_DENSE_MODEL", "intfloat/multilingual-e5-small")      # MIT, 118M params
OUT_MODEL = MODEL_DIR / "dense"
MAX_LEN = int(os.environ.get("BER_DENSE_MAXLEN", 64))
BATCH = int(os.environ.get("BER_DENSE_BATCH", 512))        # pairs per GPU per step; negatives = BATCH * world
EMB_BATCH = int(os.environ.get("BER_DENSE_EMB_BATCH", 2048))
LR = float(os.environ.get("BER_DENSE_LR", 5e-5))
EPOCHS = int(os.environ.get("BER_DENSE_EPOCHS", 1))
N_PAIRS = int(os.environ.get("BER_DENSE_PAIRS", 3_000_000))
TAU = 0.05
FWD_K = int(os.environ.get("BER_DENSE_FWD_K", 10))
REV_K = int(os.environ.get("BER_DENSE_REV_K", 3))
TOPK_ELEMS = int(os.environ.get("BER_TOPK_ELEMS", 1 << 30))  # score-matrix budget per GPU step (x2 bytes)
PREFIX = "query: "                                           # e5 convention; symmetric task -> same prefix


class Encoder(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        from transformers import AutoModel
        # no pooler: its weights would get no gradient (mean pooling is used) and DDP fails on unused params
        self.m = load_model(AutoModel, name, add_pooling_layer=False)

    def forward(self, input_ids, attention_mask):
        h = self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return F.normalize(((h * m).sum(1) / m.sum(1).clamp(min=1)).float(), dim=-1)   # mean pooling


class SingleMaker:
    """batch of row indices -> tokenised texts (runs inside DataLoader workers)."""

    def __init__(self, tok, texts: TextTable):
        self.tok, self.texts = tok, texts

    def __call__(self, idx):
        enc = self.tok([PREFIX + t for t in self.texts.get(idx)], max_length=MAX_LEN, truncation=True,
                       padding=True, pad_to_multiple_of=8, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "idx": torch.from_numpy(np.asarray(idx, np.int64))}


class PairMaker:
    """batch -> one tokenised tensor holding [anchors ; positives] (a single forward pass per step)."""

    def __init__(self, tok, ta: TextTable, tb: TextTable, gid: np.ndarray):
        self.tok, self.ta, self.tb, self.gid = tok, ta, tb, gid

    def __call__(self, idx):
        texts = [PREFIX + t for t in self.ta.get(idx)] + [PREFIX + t for t in self.tb.get(idx)]
        enc = self.tok(texts, max_length=MAX_LEN, truncation=True, padding=True, pad_to_multiple_of=8,
                       return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "gid": torch.from_numpy(self.gid[idx])}


def gather_with_grad(x: torch.Tensor, d: Dist) -> torch.Tensor:
    """All ranks' rows; the local slice keeps its autograd graph (CLIP-style local loss)."""
    if d.world == 1:
        return x
    xs = [torch.empty_like(x) for _ in range(d.world)]
    dist.all_gather(xs, x.detach().contiguous())
    xs[d.rank] = x
    return torch.cat(xs)


def gather_plain(x: torch.Tensor, d: Dist) -> torch.Tensor:
    if d.world == 1:
        return x
    xs = [torch.empty_like(x) for _ in range(d.world)]
    dist.all_gather(xs, x.contiguous())
    return torch.cat(xs)


# ---------------------------------------------------------------------------------------------- train
def training_pairs(d: Dist):
    used = pl.concat(list(s1_subset("train").values()))
    gt = (load_ground_truth().drop_nulls().rename({"source1_entity_id": "s1", "match": "cand"})
          .filter(~pl.col("s1").is_in(used.implode())))
    gt = gt.sample(min(N_PAIRS, len(gt)), seed=SEED, shuffle=True)
    s1r = records("train", [1], gt["s1"].unique(), extra=("country", "state")).rename(
        {"entity_id": "s1", "text": "ta", "state": "grp"})
    cr = records("train", [2, 3], gt["cand"].unique()).rename({"entity_id": "cand", "text": "tb"})
    pairs = (gt.join(s1r, on="s1").join(cr, on="cand")
               .with_columns(pl.col("s1").rank("dense").cast(pl.Int64).alias("gid")))
    d.log(f"dense training pairs: {len(pairs):,} (S1 outside the LightGBM train/val subsets)")
    return pairs


def grouped_batches(pairs: pl.DataFrame, seed: int):
    """Batches drawn from one (country, state) at a time -> in-batch negatives share the region (harder)."""
    rng = np.random.default_rng(seed)
    order = (pairs.select(pl.col("country"), pl.col("grp"), pl.Series("r", rng.random(len(pairs))))
                  .with_row_index("i").sort("country", "grp", "r")["i"].to_numpy())
    batches = [order[i:i + BATCH] for i in range(0, len(order) - BATCH + 1, BATCH)]
    rng.shuffle(batches)
    return batches


def contrastive_loss(e: torch.Tensor, gid: torch.Tensor, d: Dist) -> torch.Tensor:
    B = gid.shape[0]
    q, p = e[:B], e[B:]
    Q, P = gather_with_grad(q, d), gather_with_grad(p, d)
    G = gather_plain(gid, d)
    labels = torch.arange(B, device=e.device) + d.rank * B
    same = gid[:, None] == G[None, :]                        # other rows of the same S1 are not negatives
    same[torch.arange(B, device=e.device), labels] = False
    lq = F.cross_entropy((q @ P.T / TAU).masked_fill(same, float("-inf")), labels)
    lp = F.cross_entropy((p @ Q.T / TAU).masked_fill(same, float("-inf")), labels)
    return (lq + lp) / 2


def train():
    d = Dist()
    torch.manual_seed(SEED)
    pairs = training_pairs(d)
    tok = load_tokenizer(BASE)
    maker = PairMaker(tok, TextTable(pairs["ta"]), TextTable(pairs["tb"]), pairs["gid"].to_numpy())
    model = maybe_compile(Encoder(BASE).to(d.device))
    ddp = wrap_ddp(model, d)
    dtype = amp_dtype(d.device)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    n_steps = (len(pairs) // BATCH // d.world) * EPOCHS
    opt = adamw(ddp, LR)
    sched = lr_schedule(opt, n_steps)
    d.log(f"train {BASE}: {n_steps} steps, {BATCH}/GPU x {d.world} GPUs, amp {dtype}")
    step, t0 = 0, time.time()
    ddp.train()
    for ep in range(EPOCHS):
        mine = rank_share(grouped_batches(pairs, SEED + ep), d, equal=True)
        for bt in loader(mine, maker):
            bt = to_dev(bt, d.device)
            with torch.autocast(d.device.type, dtype=dtype):
                e = ddp(bt["input_ids"], bt["attention_mask"])
            loss = contrastive_loss(e, bt["gid"], d)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 100 == 0 or step == n_steps:
                rate = step * BATCH * d.world / (time.time() - t0)
                d.log(f"  ep {ep} step {step}/{n_steps} loss {loss.item():.4f} {rate:,.0f} pairs/s {gpu_mem()}")
    if d.main:
        OUT_MODEL.mkdir(parents=True, exist_ok=True)
        getattr(model, "_orig_mod", model).m.save_pretrained(OUT_MODEL)
        tok.save_pretrained(OUT_MODEL)
        d.log(f"saved {OUT_MODEL}")
    d.barrier()
    d.close()


# ---------------------------------------------------------------------------------------------- search
@torch.inference_mode()
def embed(d: Dist, model, tok, texts: TextTable) -> torch.Tensor:
    """Every rank embeds its share of length-sorted batches, then all ranks receive the full matrix (on GPU)."""
    mine = rank_share(length_batches(texts.lengths(), EMB_BATCH, shuffle=False), d, equal=False)
    idx, emb = [], []
    for bt in loader(mine, SingleMaker(tok, texts)):
        idx.append(bt["idx"])
        emb.append(model(bt["input_ids"].to(d.device, non_blocking=True),
                         bt["attention_mask"].to(d.device, non_blocking=True)).half())
    dim = model.m.config.hidden_size
    idx = torch.cat(idx).to(d.device) if idx else torch.empty(0, dtype=torch.long, device=d.device)
    emb = torch.cat(emb) if emb else torch.empty(0, dim, dtype=torch.half, device=d.device)
    full = torch.zeros(len(texts), dim, dtype=torch.half, device=d.device)
    if d.world == 1:
        full[idx] = emb
        return full
    n = torch.tensor([len(idx)], device=d.device)
    ns = [torch.empty_like(n) for _ in range(d.world)]
    dist.all_gather(ns, n)
    m = int(max(x.item() for x in ns))
    pad_i = torch.full((m,), -1, dtype=torch.long, device=d.device)
    pad_e = torch.zeros(m, dim, dtype=torch.half, device=d.device)
    pad_i[:len(idx)], pad_e[:len(idx)] = idx, emb
    all_i = [torch.empty_like(pad_i) for _ in range(d.world)]
    all_e = [torch.empty_like(pad_e) for _ in range(d.world)]
    dist.all_gather(all_i, pad_i)
    dist.all_gather(all_e, pad_e)
    for i, e in zip(all_i, all_e):
        ok = i >= 0
        full[i[ok]] = e[ok]
    return full


@torch.inference_mode()
def gpu_topk(Q: torch.Tensor, I: torch.Tensor, k: int):
    """Exact cosine top-k (rows are L2-normalised) in query blocks sized to the GPU budget."""
    k = min(k, I.shape[0])
    if Q.shape[0] == 0 or k == 0:
        return np.empty((0, k), np.float32), np.empty((0, k), np.int64)
    bs = max(32, TOPK_ELEMS // I.shape[0])
    vs, ix = [], []
    for s in range(0, Q.shape[0], bs):
        v, i = (Q[s:s + bs] @ I.T).topk(k, dim=1)
        vs.append(v.float().cpu())
        ix.append(i.cpu())
    return torch.cat(vs).numpy(), torch.cat(ix).numpy()


def _pairs_frame(q_rows, v, i, s1_is_query: bool, prefix: str) -> pl.DataFrame:
    k = v.shape[1]
    q = np.repeat(q_rows, k)
    c = i.reshape(-1)
    rk = np.tile(np.arange(1, k + 1, dtype=np.int16), len(q_rows))
    s1_row, pool_row = (q, c) if s1_is_query else (c, q)
    return pl.DataFrame({"q": s1_row.astype(np.int32), "c": pool_row.astype(np.int32),
                         f"sim_{prefix}_dense": v.reshape(-1).astype(np.float32), f"rk_{prefix}_dense": rk})


def search(split: str):
    d = Dist()
    DENSE.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(str(OUT_MODEL))
    model = Encoder(str(OUT_MODEL)).to(d.device).to(amp_dtype(d.device)).eval()
    wanted = pl.concat(list(s1_subset(split).values()))
    for country in countries(split):
        out = DENSE / f"{split}_{country}.parquet"
        if out.exists():
            d.log(f"[{split}] {country}: cached")
            continue
        t0 = time.time()
        s1 = records(split, [1], country=country)
        pool = records(split, [2, 3], country=country)
        keep = s1["entity_id"].is_in(wanted.implode()).to_numpy()
        ES = embed(d, model, tok, TextTable(s1["text"]))
        EP = embed(d, model, tok, TextTable(pool["text"]))
        d.log(f"[{split}] {country}: embedded {len(s1):,} S1 + {len(pool):,} pool in {time.time() - t0:.0f}s")
        my_q = np.array_split(np.where(keep)[0], d.world)[d.rank]
        v, i = gpu_topk(ES[torch.from_numpy(my_q).to(d.device)], EP, FWD_K)
        fwd = _pairs_frame(my_q, v, i, True, "f")
        my_p = np.array_split(np.arange(len(pool)), d.world)[d.rank]
        v, i = gpu_topk(EP[torch.from_numpy(my_p).to(d.device)], ES, REV_K)
        rev = _pairs_frame(my_p, v, i, False, "r")
        rev = rev.filter(pl.Series(keep[rev["q"].to_numpy()])) if len(rev) else rev
        fwd.write_parquet(DENSE / f"tmp_{split}_{country}_f{d.rank}.parquet")
        rev.write_parquet(DENSE / f"tmp_{split}_{country}_r{d.rank}.parquet")
        del ES, EP
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        d.barrier()
        if d.main:
            parts = {x: pl.concat([pl.read_parquet(DENSE / f"tmp_{split}_{country}_{x}{r}.parquet")
                                   for r in range(d.world)]) for x in ("f", "r")}
            m = parts["f"].join(parts["r"], on=["q", "c"], how="full", coalesce=True)
            m = m.with_columns(pl.Series("s1", s1["entity_id"].to_numpy()[m["q"].to_numpy()]),
                               pl.Series("cand", pool["entity_id"].to_numpy()[m["c"].to_numpy()])).drop("q", "c")
            m.write_parquet(out)
            for x in ("f", "r"):
                for r in range(d.world):
                    (DENSE / f"tmp_{split}_{country}_{x}{r}.parquet").unlink()
            d.log(f"[{split}] {country}: {len(m):,} dense pairs ({len(parts['f']):,} fwd, {len(parts['r']):,} rev) "
                  f"in {time.time() - t0:.0f}s {gpu_mem()}")
        d.barrier()
    if d.main and split == "train":
        report_recall()
    d.close()


def report_recall():
    """Recall of the dense leg alone and of TF-IDF + dense on the train/val subsets (before merge_dense)."""
    from pipeline import CAND
    gt = load_ground_truth().drop_nulls().rename({"source1_entity_id": "s1", "match": "cand"})
    dense = pl.concat([pl.read_parquet(f, columns=["s1", "cand"]) for f in DENSE.glob("train_*.parquet")])
    for name, ids in s1_subset("train").items():
        g = gt.filter(pl.col("s1").is_in(ids.implode()))
        dn = g.join(dense, on=["s1", "cand"]).height
        msg = f"  {name}: dense-leg recall {dn / len(g):.4f}"
        f = CAND / f"{name}.parquet"
        if f.exists():
            tf = pl.read_parquet(f, columns=["s1", "cand"])
            both = pl.concat([tf, dense]).unique()
            msg += (f" | TF-IDF {g.join(tf, on=['s1', 'cand']).height / len(g):.4f}"
                    f" | union {g.join(both, on=['s1', 'cand']).height / len(g):.4f}")
        print(msg, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "search":
        search(sys.argv[2])
