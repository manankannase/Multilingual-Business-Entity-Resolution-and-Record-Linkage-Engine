"""Same XLM-R pair classifier; conservative pseudo weighting and recoverable training."""
import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from pair_data import prepare


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_save(state, path):
    tmp = path.with_suffix(".partial")
    torch.save(state, tmp)
    os.replace(tmp, path)


class Pairs(Dataset):
    def __init__(self, df, pseudo_weight):
        self.a, self.b, self.y = (df[k].to_list() for k in ("a", "b", "label"))
        self.w = np.where(df["origin"].to_numpy() == "train", 1., pseudo_weight)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.a[i], self.b[i], self.y[i], self.w[i]


class Collator:
    def __init__(self, tok, length):
        self.tok, self.length = tok, length
    def __call__(self, rows):
        a, b, y, w = zip(*rows)
        enc = self.tok(list(a), list(b), truncation=True, max_length=self.length,
                       padding=True, return_tensors="pt")
        return enc, torch.tensor(y, dtype=torch.long), torch.tensor(w, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pairs_fr.parquet")
    ap.add_argument("--out", default="ce_xlmr_improved")
    ap.add_argument("--model", default="FacebookAI/xlm-roberta-large")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--pseudo_weight", type=float, default=.35)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--cpu", action="store_true", help="tiny local test only")
    ap.add_argument("--no_gradient_checkpointing", action="store_true")
    args = ap.parse_args()
    if min(args.batch, args.accum, args.eval_batch, args.ckpt_every) < 1 or not 0 <= args.pseudo_weight <= 1:
        ap.error("positive batch/accum/checkpoint interval and pseudo_weight in [0,1] required")
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; install CUDA PyTorch and select your assigned GPU")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # Prevent two processes writing the same model/checkpoint.
    import fcntl
    lock = open(out / ".lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = dict(vars(args), pairs_sha256=digest(args.pairs),
                  code_sha256=digest(__file__) + digest(Path(__file__).with_name("pair_data.py")),
                  torch=torch.__version__)
    manifest = out / "run_config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("Configuration/input/code changed: use a NEW output directory")
    manifest.write_text(json.dumps(config, indent=2))
    if (out / "COMPLETE.json").exists():
        print("Already complete:", out, flush=True); return
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cpu" if args.cpu else "cuda")
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    amp = lambda: torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else contextlib.nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not bf16)
    fit, dev, audit, report = prepare(args.pairs, args.limit)
    (out / "data_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    collate = Collator(tok, args.max_len)
    data = Pairs(fit, args.pseudo_weight)
    order = np.random.default_rng(args.seed).permutation(len(data)).tolist()
    micros = math.ceil(len(data)/args.batch)
    steps = math.ceil(micros/args.accum)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = get_linear_schedule_with_warmup(opt, int(.06*steps), steps)
    ck = out / "state.pt"
    start = 0
    if ck.exists():
        # Only load our own trusted local checkpoint.
        state = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
        start = state["micro"]
        torch.set_rng_state(state["rng"])
        if device.type == "cuda": torch.cuda.set_rng_state_all(state["cuda_rng"])
        print("Resumed at microbatch", start, flush=True)
    loader = DataLoader(Subset(data, order[start*args.batch:]), batch_size=args.batch,
                        collate_fn=collate, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    model.train(); opt.zero_grad(set_to_none=True)
    t0 = time.monotonic()
    for micro, (enc, labels, weights) in enumerate(loader, start):
        enc = {k:v.to(device) for k,v in enc.items()}
        labels, weights = labels.to(device), weights.to(device)
        # Normalize by real pair count in the effective batch, including final partial batch.
        window = (micro//args.accum)*args.accum*args.batch
        count = min(args.accum*args.batch, len(data)-window)
        with amp():
            logits = model(**enc).logits
            loss = (torch.nn.functional.cross_entropy(logits.float(), labels, reduction="none")*weights).sum()/count
        if not torch.isfinite(loss): raise RuntimeError("Non-finite training loss")
        scaler.scale(loss).backward()
        boundary = (micro+1)%args.accum == 0 or micro+1 == micros
        if not boundary: continue
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        old_scale = scaler.get_scale()
        scaler.step(opt); scaler.update()
        if scaler.get_scale() >= old_scale: scheduler.step()
        opt.zero_grad(set_to_none=True)
        step = math.ceil((micro+1)/args.accum)
        if step%100 == 0 or micro+1 == micros:
            eta = (micros-micro-1)*(time.monotonic()-t0)/max(1,micro+1-start)/60
            print(f"step {step}/{steps} loss_last_micro={loss.item():.5f} ETA={eta:.1f}min", flush=True)
        if step%args.ckpt_every == 0 or micro+1 == micros:
            atomic_save(dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                             scaler=scaler.state_dict(), micro=micro+1, rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else []), ck)
    model.eval()
    def evaluate(frame):
        probs=[]; labels=[]
        dl=DataLoader(Pairs(frame,1),batch_size=args.eval_batch,collate_fn=collate)
        with torch.inference_mode():
            for enc,y,_ in dl:
                with amp(): logits=model(**{k:v.to(device) for k,v in enc.items()}).logits.float()
                probs.extend(logits.softmax(-1)[:,1].cpu().tolist()); labels.extend(y.tolist())
        p=np.clip(np.array(probs),1e-7,1-1e-7); y=np.array(labels)
        return {"pairs":len(y),"accuracy_at_0.5":float(((p>=.5)==y).mean()),
                "log_loss":float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()),
                "note":"True-label pair sanity check; NOT entity macro F0.5 or France accuracy."}
    scores={"dev":evaluate(dev),"audit":evaluate(audit)}
    model.save_pretrained(out); tok.save_pretrained(out)
    (out/"COMPLETE.json").write_text(json.dumps(scores,indent=2))
    print(json.dumps(scores,indent=2),flush=True)
    print("Saved model:",out,"Checkpoint retained for recovery.",flush=True)


if __name__ == "__main__": main()
