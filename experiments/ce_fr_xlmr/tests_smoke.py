"""Offline tiny XLM-R execution test, including interrupted/resumed training parity."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import polars as pl
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaForSequenceClassification
from pair_data import prepare


def main():
    root=Path(__file__).resolve().parent
    tmp=Path(tempfile.mkdtemp(prefix="ce-fr-smoke-"))
    base=tmp/"tiny-model"; base.mkdir()
    vocab={"<s>":0,"<pad>":1,"</s>":2,"<unk>":3,"name":4,"address":5,"shop":6,"street":7}
    tokenizer=Tokenizer(WordLevel(vocab,unk_token="<unk>"))
    tokenizer.pre_tokenizer=Whitespace()
    tokenizer.post_processor=TemplateProcessing(single="<s> $A </s>",pair="<s> $A </s> </s> $B </s>",special_tokens=[("<s>",0),("</s>",2)])
    fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,bos_token="<s>",eos_token="</s>",unk_token="<unk>",pad_token="<pad>",model_input_names=["input_ids","attention_mask"])
    fast.save_pretrained(base)
    torch.manual_seed(1)
    model=XLMRobertaForSequenceClassification(XLMRobertaConfig(vocab_size=len(vocab),hidden_size=16,
        intermediate_size=32,num_attention_heads=2,num_hidden_layers=1,max_position_embeddings=256,num_labels=2))
    model.save_pretrained(base)
    pairs=tmp/"pairs.parquet"
    pl.DataFrame({"a":[f"name shop {i} | address {i}" for i in range(4000)],
                  "b":[f"name target {i} | street {i}" for i in range(4000)],
                  "label":[i%2 for i in range(4000)],
                  "origin":["fr_pseudo" if i%5==0 else "train" for i in range(4000)]}).write_parquet(pairs)
    fit,dev,audit,_=prepare(pairs)
    assert not set(fit["a"]) & (set(dev["a"]) | set(audit["a"]))
    assert not set(fit["b"]) & (set(dev["b"]) | set(audit["b"]))
    assert set(dev["origin"])==set(audit["origin"])=={"train"}
    env=dict(os.environ,OMP_NUM_THREADS="1",TOKENIZERS_PARALLELISM="false",HF_HUB_OFFLINE="1",PYTHONPATH=str(root))
    common=["--pairs",str(pairs),"--model",str(base),"--cpu","--limit","32","--batch","4","--accum","2","--eval_batch","16","--ckpt_every","1"]
    full=tmp/"full"; resumed=tmp/"resumed"
    subprocess.run([sys.executable,str(root/"train_ce.py"),*common,"--out",str(full)],env=env,check=True)
    wrapper="import train_ce; old=train_ce.atomic_save\ndef interrupted(*a):\n old(*a)\n raise SystemExit(99)\ntrain_ce.atomic_save=interrupted\ntrain_ce.main()"
    r=subprocess.run([sys.executable,"-c",wrapper,*common,"--out",str(resumed)],env=env)
    assert r.returncode==99
    subprocess.run([sys.executable,str(root/"train_ce.py"),*common,"--out",str(resumed)],env=env,check=True)
    a=XLMRobertaForSequenceClassification.from_pretrained(full).state_dict()
    b=XLMRobertaForSequenceClassification.from_pretrained(resumed).state_dict()
    assert all(torch.equal(a[k],b[k]) for k in a), "Resume changed weights"
    subprocess.run([sys.executable,str(root/"score_pairs.py"),"--model",str(full),"--pairs",str(pairs),"--out",str(tmp/"scores.parquet"),"--batch","128"],env=env,check=True)
    scores=pl.read_parquet(tmp/"scores.parquet")
    assert scores.height==4000 and scores["ce_probability"].is_between(0,1).all()
    print("PASS: grouped splits, CPU tiny XLM-R training, exact resumed weights, scoring.",tmp)


if __name__=="__main__": main()
