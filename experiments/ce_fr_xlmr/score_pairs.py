"""Score a candidate parquet with a,b; retains optional source IDs for downstream matching."""
import argparse
from pathlib import Path
import contextlib
import polars as pl
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--pairs",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--batch",type=int,default=16)
    args=ap.parse_args()
    if args.batch<1: ap.error("batch must be positive")
    model_dir=Path(args.model)
    if not (model_dir/"COMPLETE.json").exists(): raise ValueError("Training not complete")
    out=Path(args.out)
    if out.exists(): raise FileExistsError("Choose a new output path")
    df=pl.read_parquet(args.pairs)
    if not {"a","b"} <= set(df.columns): raise ValueError("Expected a,b pair text")
    if any(df.select("a","b").null_count().row(0)): raise ValueError("Null pair text")
    import json
    length=json.loads((model_dir/"run_config.json").read_text())["max_len"]
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    tok=AutoTokenizer.from_pretrained(model_dir)
    dtype=torch.bfloat16 if device=="cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scores=[]
    with torch.inference_mode():
        for start in range(0,df.height,args.batch):
            batch=df.slice(start,args.batch)
            enc=tok(batch["a"].to_list(),batch["b"].to_list(),padding=True,truncation=True,max_length=length,return_tensors="pt")
            amp=torch.autocast("cuda",dtype=dtype) if device=="cuda" else contextlib.nullcontext()
            with amp: logits=model(**{k:v.to(device) for k,v in enc.items()}).logits.float()
            scores.extend(logits.softmax(-1)[:,1].cpu().tolist())
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix(out.suffix+".partial")
    df.with_columns(pl.Series("ce_probability",scores,dtype=pl.Float32)).write_parquet(tmp)
    tmp.replace(out)
    print("Saved candidate scores:",out,"These are not a submission TSV.")


if __name__=="__main__": main()
