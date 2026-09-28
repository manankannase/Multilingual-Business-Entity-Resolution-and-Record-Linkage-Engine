"""Fixed one-epoch continuation with recoverable checkpoints and sharded inference."""
import argparse
import contextlib
import fcntl
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import KEYS, atomic_json, atomic_parquet, code_digest, digest, verify_pairs


def match_logit(logits):
    if logits.ndim != 2 or logits.shape[1] not in (1, 2):
        raise ValueError('Expected one-logit or two-logit binary classifier')
    return logits[:, 0] if logits.shape[1] == 1 else logits[:, 1] - logits[:, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['check', 'train', 'score'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--fold', type=int, choices=[0, 1], default=0)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--checkpoint-every', type=int, default=250)
    parser.add_argument('--score-chunk', type=int, default=10000)
    parser.add_argument('--cpu', action='store_true', help='Local tiny-model test only')
    args = parser.parse_args()
    if min(args.batch, args.accum, args.checkpoint_every, args.score_chunk) < 1 or args.lr <= 0 or args.max_length < 16:
        parser.error('Invalid training or scoring settings')
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in GPU Python environment')
    if not args.cpu and torch.cuda.device_count() != 1:
        raise RuntimeError('Select exactly one assigned GPU with CUDA_VISIBLE_DEVICES')
    device = torch.device('cpu' if args.cpu else 'cuda')
    if args.mode == 'check':
        print('GPU_RUNTIME_PASS', torch.__version__, torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU', flush=True)
        return
    out = Path(args.out).resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    if manifest['code_sha256'] != code_digest():
        raise ValueError('Prepared data/code mismatch')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    amp = lambda: torch.autocast('cuda', dtype=dtype) if device.type == 'cuda' else contextlib.nullcontext()
    folder = out / f'model_fold{args.fold}'
    folder.mkdir(exist_ok=True)
    lock = (folder / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.mode == 'train':
        path = out / f'fit_fold{args.fold}.parquet'
        config = {k: getattr(args, k) for k in ['fold', 'batch', 'accum', 'lr', 'max_length', 'checkpoint_every', 'cpu']}
        config.update(data_sha256=digest(path), code_sha256=code_digest(),
                      initial_checkpoint=manifest['checkpoints'][args.fold])
        if (folder / 'run.json').exists() and json.loads((folder / 'run.json').read_text()) != config:
            raise ValueError('Resume requires unchanged input/settings')
        atomic_json(folder / 'run.json', config)
        if (folder / 'COMPLETE.json').exists():
            print(f'Fold {args.fold} already complete', flush=True)
            return
        frame = pl.read_parquet(path)
        verify_pairs(frame, labels=True)
        if any(frame.select('a', 'b', 'weight').null_count().row(0)):
            raise ValueError('Null training texts/weights')
        source = config['initial_checkpoint']
    else:
        if not (folder / 'COMPLETE.json').exists():
            raise ValueError('Fine-tuning incomplete')
        config = json.loads((folder / 'run.json').read_text())
        if args.max_length != config['max_length']:
            raise ValueError('Scoring token limit must match training')
        source = str(folder)
    # Local checkpoints/tokenizers only: no model-name guess or network download.
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(source, local_files_only=True).to(device)
    def encode(data):
        return {k: v.to(device) for k, v in tokenizer(data['a'].to_list(), data['b'].to_list(),
            truncation='longest_first', max_length=args.max_length,
            padding=True, return_tensors='pt').items()}
    if args.mode == 'score':
        path = out / f'{args.split}_pairs.parquet'
        score_dir = out / f'{args.split}_scores_fold{args.fold}'
        score_dir.mkdir(exist_ok=True)
        score_manifest = dict(input_sha256=digest(path), training=config,
                              chunk=args.score_chunk, batch=args.batch, split=args.split)
        if (score_dir / 'manifest.json').exists() and json.loads((score_dir / 'manifest.json').read_text()) != score_manifest:
            raise ValueError('Scoring input/settings changed')
        atomic_json(score_dir / 'manifest.json', score_manifest)
        frame = pl.read_parquet(path)
        verify_pairs(frame)
        model.eval()
        start_time, done_now = time.monotonic(), 0
        with torch.inference_mode():
            for offset in range(0, frame.height, args.score_chunk):
                target = score_dir / f'{offset:012d}.parquet'
                block = frame.slice(offset, args.score_chunk)
                if target.exists():
                    saved = pl.read_parquet(target)
                    if not saved.select(KEYS).equals(block.select(KEYS)) or not saved['ft_logit'].is_finite().all():
                        raise ValueError(f'Invalid cached scoring shard {target}')
                    continue
                scores = []
                for index in range(0, block.height, args.batch):
                    batch = block.slice(index, args.batch)
                    with amp():
                        value = match_logit(model(**encode(batch)).logits).float()
                    if not torch.isfinite(value).all():
                        raise RuntimeError('Nonfinite inference logit')
                    scores.extend(value.cpu().tolist())
                    done_now += batch.height
                    if index // args.batch % 100 == 0:
                        rate = done_now / max(time.monotonic() - start_time, 1e-6)
                        eta = (frame.height - offset - index - batch.height) / max(rate, 1e-6) / 60
                        print(f'SCORE {args.split} fold={args.fold} pairs={offset+index+batch.height:,}/{frame.height:,} rate={rate:.1f}/s ETA={eta:.1f}min', flush=True)
                atomic_parquet(block.select(KEYS).with_columns(
                    pl.Series('ft_logit', scores, dtype=pl.Float32)), target)
        atomic_json(score_dir / 'COMPLETE.json', dict(pairs=frame.height))
        print(f'SCORE_COMPLETE {args.split} fold={args.fold}', flush=True)
        return
    seed = 42 + args.fold
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    batches = math.ceil(frame.height / args.batch)
    steps = math.ceil(batches / args.accum)
    warmup = max(1, int(.06 * steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda s: (s + 1) / warmup if s < warmup else max(0., (steps - s) / max(1, steps - warmup)))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and not bf16)
    order = np.random.default_rng(seed).permutation(frame.height)
    first = 0
    checkpoint = folder / 'state.pt'
    if checkpoint.exists():
        # Only this package's own trusted recovery checkpoint is deserialized.
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        first = state['next_batch']
        torch.set_rng_state(state['rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
        print(f'RESUMED fold={args.fold} batch={first}/{batches}', flush=True)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    print(f'TRAIN_START fold={args.fold} pairs={frame.height:,} steps={steps} fixed_epochs=1', flush=True)
    for batch_id in range(first, batches):
        indices = order[batch_id * args.batch:(batch_id + 1) * args.batch]
        batch = frame[indices.tolist()]
        labels = torch.tensor(batch['y'].to_list(), dtype=torch.float32, device=device)
        weights = torch.tensor(batch['weight'].to_list(), dtype=torch.float32, device=device)
        window = (batch_id // args.accum) * args.accum * args.batch
        count = min(args.accum * args.batch, frame.height - window)
        with amp():
            logits = match_logit(model(**encode(batch)).logits).float()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, labels,
                     reduction='none') * weights).sum() / count
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        scaler.scale(loss).backward()
        boundary = (batch_id + 1) % args.accum == 0 or batch_id + 1 == batches
        if not boundary:
            continue
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= previous_scale:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step = math.ceil((batch_id + 1) / args.accum)
        if step % 25 == 0 or batch_id + 1 == batches:
            rate = (batch_id + 1 - first) * args.batch / max(time.monotonic() - started, 1e-6)
            eta = (batches - batch_id - 1) * args.batch / max(rate, 1e-6) / 60
            print(f'TRAIN fold={args.fold} step={step}/{steps} loss_micro={loss.item():.5f} rate={rate:.1f}/s ETA={eta:.1f}min', flush=True)
        if step % args.checkpoint_every == 0 or batch_id + 1 == batches:
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                next_batch=batch_id + 1, rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
            torch.save(state, folder / 'state.pt.partial')
            os.replace(folder / 'state.pt.partial', checkpoint)
            print(f'CHECKPOINT_SAVED fold={args.fold} step={step}', flush=True)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    atomic_json(folder / 'COMPLETE.json', dict(pairs=frame.height, steps=steps, fixed_epochs=1))
    print(f'TRAIN_COMPLETE fold={args.fold}', flush=True)


if __name__ == '__main__':
    main()
