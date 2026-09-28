"""GPU cross-encoder re-ranker: reads (S1 text, candidate text) jointly and outputs a match logit, which stage 2
uses as a feature (plus its rank / margin inside the candidate graph).

  python stage2.py oof                                     # (CPU) stage-1 OOF scores + state folds
  torchrun --nproc_per_node=G gpu_xenc.py train 0          # model trained on fold-1 states
  torchrun --nproc_per_node=G gpu_xenc.py train 1          # model trained on fold-0 states
  torchrun --nproc_per_node=G gpu_xenc.py score train      # each train pair scored by the model that never saw it
  torchrun --nproc_per_node=G gpu_xenc.py score val        # val / test: mean logit of both fold models
  torchrun --nproc_per_node=G gpu_xenc.py score test
  python stage2.py fit && python stage2.py predict

Only pairs with stage-1 p >= P_MIN are used (same set stage 2 sees), so training focuses on the hard pairs and
test scoring touches ~15-25% of the 75M candidates.

The same script trains a second, decoder-LLM judge (BER_XENC_TAG=llm, BER_XENC_MODEL=Qwen/Qwen2.5-1.5B,
Apache-2.0): a sequence-classification head on the last token of a short prompt, restricted to the uncertain
band BER_XENC_P_LO <= p < BER_XENC_P_HI. Outputs go to xenc/<tag>_<split>.parquet, column <tag>.
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from config import SEED, WORK_DIR
from gpu_common import (MODEL_DIR, Dist, TextTable, adamw, amp_dtype, gpu_mem, length_batches, load_model,
                        load_tokenizer, loader, lr_schedule, maybe_compile, rank_share, records, to_dev,
                        wrap_ddp)

BASE = os.environ.get("BER_XENC_MODEL", "intfloat/multilingual-e5-base")        # MIT, 278M params
MAX_LEN = int(os.environ.get("BER_XENC_MAXLEN", 128))
BATCH = int(os.environ.get("BER_XENC_BATCH", 256))           # training pairs per GPU per step
INF_BATCH = int(os.environ.get("BER_XENC_INF_BATCH", 2048))
LR = float(os.environ.get("BER_XENC_LR", 3e-5))
EPOCHS = int(os.environ.get("BER_XENC_EPOCHS", 2))
P_MIN = float(os.environ.get("BER_P_MIN", 0.01))
TAG = os.environ.get("BER_XENC_TAG", "xenc")
P_LO = float(os.environ.get("BER_XENC_P_LO", P_MIN))
P_HI = float(os.environ.get("BER_XENC_P_HI", 1.01))
GRAD_CKPT = os.environ.get("BER_GRAD_CKPT", "0") == "1"
DECODER_TYPES = {"qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2", "phi3", "olmo2", "granite"}
XDIR = WORK_DIR / "xenc"
SOURCES = {"train": ("train", WORK_DIR / "s2_train_p.parquet"),
           "val": ("train", WORK_DIR / "s2_val_p.parquet"),
           "test": ("test", WORK_DIR / "test_scored.parquet")}


def is_decoder(name: str) -> bool:
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(name).model_type in DECODER_TYPES


def model_dir(k) -> "Path":
    return MODEL_DIR / f"{TAG}_fold{k}"


class PairMaker:
    def __init__(self, tok, ta: TextTable, tb: TextTable, ia, ib, y=None, prompt: bool = False):
        self.tok, self.ta, self.tb, self.ia, self.ib, self.y = tok, ta, tb, ia, ib, y
        self.prompt = prompt

    def __call__(self, idx):
        A, B = self.ta.get(self.ia[idx]), self.tb.get(self.ib[idx])
        if self.prompt:           # decoder LLM: one prompt, classification head reads the last token
            texts = [f"Record A: {a[:300]}\nRecord B: {b[:300]}\nSame business? Answer:" for a, b in zip(A, B)]
            enc = self.tok(texts, truncation=True, max_length=MAX_LEN, padding=True, pad_to_multiple_of=8,
                           return_tensors="pt")
        else:
            enc = self.tok(A, B, truncation="longest_first", max_length=MAX_LEN, padding=True,
                           pad_to_multiple_of=8, return_tensors="pt")
        out = {k: enc[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc}
        out["idx"] = torch.from_numpy(np.asarray(idx, np.int64))
        if self.y is not None:
            out["y"] = torch.from_numpy(self.y[idx])
        return out


class PairTable:
    """Pairs of one split with row indices into two packed text tables (each record's text stored once)."""

    def __init__(self, name: str):
        data_split, path = SOURCES[name]
        self.pairs = pl.read_parquet(path).filter((pl.col("p") >= max(P_MIN, P_LO)) & (pl.col("p") < P_HI))
        s1r = records(data_split, [1], self.pairs["s1"].unique()).with_row_index("ia")
        cr = records(data_split, [2, 3], self.pairs["cand"].unique()).with_row_index("ib")
        self.pairs = (self.pairs.join(s1r.select(pl.col("entity_id").alias("s1"), "ia"), on="s1", how="left")
                                .join(cr.select(pl.col("entity_id").alias("cand"), "ib"), on="cand", how="left")
                                .drop_nulls(["ia", "ib"]))
        self.ta, self.tb = TextTable(s1r["text"]), TextTable(cr["text"])
        self.ia = self.pairs["ia"].to_numpy().astype(np.int64)
        self.ib = self.pairs["ib"].to_numpy().astype(np.int64)
        self.y = self.pairs["y"].to_numpy().astype(np.float32) if "y" in self.pairs.columns else None
        self.lengths = self.ta.lengths()[self.ia] + self.tb.lengths()[self.ib]

    def __len__(self):
        return len(self.pairs)

    def maker(self, tok, with_y: bool, prompt: bool = False):
        return PairMaker(tok, self.ta, self.tb, self.ia, self.ib, self.y if with_y else None, prompt)


def load_tok(name: str):
    tok = load_tokenizer(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_clf(name: str, tok, **kw):
    from transformers import AutoModelForSequenceClassification
    m = load_model(AutoModelForSequenceClassification, name, **kw)
    m.config.pad_token_id = tok.pad_token_id      # decoders: head reads the last non-pad token
    return m


def _fwd(model, bt):
    kw = {k: bt[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in bt}
    return model(**kw).logits.squeeze(-1).float()


@torch.inference_mode()
def infer(d: Dist, model, tok, T: PairTable, rows: np.ndarray, tag: str, prompt: bool) -> np.ndarray:
    """Logits for T rows `rows` (all ranks share the work; result returned on every rank)."""
    batches = [rows[b] for b in length_batches(T.lengths[rows], INF_BATCH, shuffle=False)]
    mine = rank_share(batches, d, equal=False)
    idx, out = [], []
    t0 = time.time()
    for j, bt in enumerate(loader(mine, T.maker(tok, False, prompt))):
        idx.append(bt["idx"].numpy())
        out.append(_fwd(model, to_dev(bt, d.device)).cpu().numpy())
        if j % 500 == 0 and j:
            done = sum(len(x) for x in idx) * d.world
            d.log(f"  {tag}: ~{done:,}/{len(rows):,} pairs, {done / (time.time() - t0):,.0f} pairs/s {gpu_mem()}")
    idx = np.concatenate(idx) if idx else np.empty(0, np.int64)
    out = np.concatenate(out) if out else np.empty(0, np.float32)
    XDIR.mkdir(parents=True, exist_ok=True)
    tmp = lambda r: XDIR / f"tmp_{TAG}_{tag}_{r}.npz"        # TAG: xenc and llm jobs may run concurrently
    np.savez(tmp(d.rank), idx=idx, out=out)
    d.barrier()
    res = np.full(len(T), np.nan, np.float32)
    for r in range(d.world):
        z = np.load(tmp(r))
        res[z["idx"]] = z["out"]
    d.barrier()
    os.remove(tmp(d.rank))
    return res


def _metrics(logit: np.ndarray, y: np.ndarray) -> str:
    p = 1 / (1 + np.exp(-logit.astype(np.float64)))
    ll = -np.mean(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1)))
    acc = np.mean((p >= 0.5) == (y == 1))
    return f"logloss {ll:.4f} acc {acc:.4f} (pos rate {y.mean():.3f})"


def train(fold: int):
    d = Dist()
    torch.manual_seed(SEED + fold)
    prompt = is_decoder(BASE)
    T = PairTable("train")
    rows = np.where(T.pairs["fold"].to_numpy() != fold)[0]
    V = PairTable("val")
    v_rows = np.random.default_rng(SEED).permutation(len(V))[:200_000]
    tok = load_tok(BASE)
    model = load_clf(BASE, tok, num_labels=1)
    if GRAD_CKPT:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = maybe_compile(model.to(d.device))
    ddp = wrap_ddp(model, d)
    dtype = amp_dtype(d.device)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    n_steps = (len(rows) // BATCH // d.world) * EPOCHS
    opt = adamw(ddp, LR)
    sched = lr_schedule(opt, n_steps, warmup=0.06)
    d.log(f"{TAG} fold {fold}: {BASE} ({'decoder prompt' if prompt else 'encoder pair'}), band p in "
          f"[{max(P_MIN, P_LO)}, {P_HI}), {len(rows):,} train pairs (pos {T.y[rows].mean():.3f}), "
          f"{n_steps} steps, {BATCH}/GPU x {d.world} GPUs, amp {dtype}")
    maker = T.maker(tok, True, prompt)
    step, t0 = 0, time.time()
    for ep in range(EPOCHS):
        ddp.train()
        batches = [rows[b] for b in length_batches(T.lengths[rows], BATCH, shuffle=True, seed=SEED + 97 * ep + fold)]
        for bt in loader(rank_share(batches, d, equal=True), maker):
            bt = to_dev(bt, d.device)
            with torch.autocast(d.device.type, dtype=dtype):
                logit = _fwd(ddp, bt)
            loss = F.binary_cross_entropy_with_logits(logit, bt["y"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 200 == 0 or step == n_steps:
                rate = step * BATCH * d.world / (time.time() - t0)
                d.log(f"  ep {ep} step {step}/{n_steps} loss {loss.item():.4f} {rate:,.0f} pairs/s {gpu_mem()}")
        ddp.eval()
        with torch.autocast(d.device.type, dtype=dtype):
            vl = infer(d, getattr(model, "_orig_mod", model), tok, V, v_rows, f"val_f{fold}_e{ep}", prompt)
        d.log(f"  epoch {ep} val: {_metrics(vl[v_rows], V.y[v_rows])}")
    if d.main:
        out = model_dir(fold)
        out.mkdir(parents=True, exist_ok=True)
        getattr(model, "_orig_mod", model).save_pretrained(out)
        tok.save_pretrained(out)
        d.log(f"saved {out}")
    d.barrier()
    d.close()


def score(name: str):
    d = Dist()
    T = PairTable(name)
    folds = sorted(int(p.name.rsplit("fold", 1)[1]) for p in MODEL_DIR.glob(f"{TAG}_fold*"))
    d.log(f"score {TAG} {name}: {len(T):,} pairs with models of folds {folds}")
    dtype = amp_dtype(d.device)
    total = np.zeros(len(T), np.float32)
    count = np.zeros(len(T), np.float32)
    for k in folds:
        path = str(model_dir(k))
        prompt = is_decoder(path)
        tok = load_tok(path)
        model = maybe_compile(load_clf(path, tok).to(d.device).to(dtype).eval())
        # train pairs: only the fold this model never saw; val/test: every pair (fold models are averaged)
        rows = np.where(T.pairs["fold"].to_numpy() == k)[0] if name == "train" else np.arange(len(T))
        res = infer(d, model, tok, T, rows, f"{name}_f{k}", prompt)
        total[rows] += res[rows]
        count[rows] += 1
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if d.main:
        x = np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)
        out = T.pairs.select("s1", "cand").with_columns(pl.Series(TAG, x))
        path = XDIR / f"{TAG}_{name}.parquet"
        out.write_parquet(path)
        msg = f"wrote {path}: {len(out):,} pairs"
        if T.y is not None:
            msg += " | " + _metrics(x, T.y)
        d.log(msg)
    d.barrier()
    d.close()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train(int(sys.argv[2]))
    elif cmd == "score":
        score(sys.argv[2])
