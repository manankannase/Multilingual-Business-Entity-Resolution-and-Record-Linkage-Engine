import argparse
import fcntl
import json
from pathlib import Path

import polars as pl

from common import (KEYS, atomic_json, atomic_parquet, attrs, code_digest,
                    digest, repair_scope, texts, verify_pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight', 'prepare', 'test'])
    parser.add_argument('--work', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-pairs', type=int, default=300000)
    args = parser.parse_args()
    work, out = Path(args.work).resolve(), Path(args.out).resolve()
    if out == work or work in out.parents or out in work.parents:
        raise ValueError('Run directory must be separate from baseline work')
    if args.max_pairs < 1000:
        raise ValueError('max-pairs must be at least 1000')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    required = ['s2_train_p.parquet', 'val_scored_s2.parquet', 'val_pred_s2.parquet',
                'cand/val.parquet', 'lgb_s2.json', 'raw/train_ground_truth_long.parquet']
    required += [f'norm/train_source{s}.parquet' for s in (1, 2, 3)]
    for name in required:
        if not (work / name).is_file():
            raise FileNotFoundError(work / name)
    for fold in (0, 1):
        model = work / 'models' / f'xenc_fold{fold}'
        if not (model / 'config.json').is_file() or not any(model.glob('*.safetensors')) and not any(model.glob('pytorch_model*.bin')):
            raise FileNotFoundError(f'Missing saved cross-encoder: {model}')
        config = json.loads((model / 'config.json').read_text())
        if config.get('num_labels', len(config.get('id2label', {}))) not in (1, 2):
            raise ValueError(f'Unsupported classifier head: {model}')
    manifest = dict(work=str(work), max_pairs=args.max_pairs,
        code_sha256=code_digest(), baseline_meta_sha256=digest(work / 'lgb_s2.json'),
        checkpoints=[str(work / 'models' / f'xenc_fold{k}') for k in (0, 1)],
        checkpoint_configs=[digest(work / 'models' / f'xenc_fold{k}' / 'config.json') for k in (0, 1)],
        sources={name: dict(bytes=(work / name).stat().st_size,
                           mtime_ns=(work / name).stat().st_mtime_ns) for name in required})
    manifest['checkpoint_files'] = {str(path): dict(bytes=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns) for fold in (0, 1)
        for path in sorted((work / 'models' / f'xenc_fold{fold}').iterdir()) if path.is_file()}
    if (out / 'manifest.json').exists() and json.loads((out / 'manifest.json').read_text()) != manifest:
        raise ValueError('Input/code changed; use a new run directory')
    if args.mode == 'preflight':
        print('PREFLIGHT_PASS: baseline caches and both saved checkpoints exist.', flush=True)
        return
    if args.mode == 'test':
        report = json.loads((out / 'comparison.json').read_text())
        if not report['eligible_for_test_prediction']:
            raise ValueError('Validation did not pass: test scoring disabled')
        if report['code_sha256'] != code_digest():
            raise ValueError('Code changed since validation')
        if (out / 'test_scope.parquet').exists():
            print('Test scope already prepared', flush=True)
            return
        pairs = pl.read_parquet(work / 'test_scored_s2.parquet').select(KEYS + ['p2'])
        verify_pairs(pairs)
        records = attrs(work, 'test', pairs)
        prepared = repair_scope(texts(pairs, records))
        if not prepared['repair'].any():
            raise ValueError('Test repair scope is empty')
        atomic_parquet(prepared.select(KEYS + ['p2', 'blank', 'repair']), out / 'test_scope.parquet')
        atomic_parquet(prepared.filter(pl.col('repair')).select(KEYS + ['a', 'b']), out / 'test_pairs.parquet')
        print(f'TEST_REPAIR_PAIRS={prepared["repair"].sum():,}/{prepared.height:,}', flush=True)
        return
    atomic_json(out / 'manifest.json', manifest)
    if (out / 'PREPARE_COMPLETE.json').exists():
        print('Preparation already complete', flush=True)
        return
    val = pl.read_parquet(work / 'val_scored_s2.parquet')
    train = pl.read_parquet(work / 's2_train_p.parquet').filter(pl.col('p') >= .01)
    verify_pairs(train, labels=True)
    verify_pairs(val, labels=True)
    if 'fold' not in train.columns or set(train['fold'].unique()) != {0, 1}:
        raise ValueError('Expected original two-fold OOF assignment')
    all_val = pl.read_parquet(work / 'cand/val.parquet', columns=['s1'])['s1'].unique()
    if set(train['s1']) & set(all_val):
        raise ValueError('Training and validation Source-1 IDs overlap')
    # Additional fine-tuning excludes every target ID appearing in validation.
    train = train.filter(~pl.col('cand').is_in(val['cand'].unique().implode()))
    if not train.height:
        raise ValueError('No training pairs after validation-target exclusion')
    val_text = repair_scope(texts(val, attrs(work, 'train', val)))
    if not val_text['repair'].any():
        raise ValueError('Validation repair scope is empty')
    atomic_parquet(val_text.select(KEYS + ['p', 'p2', 'y', 'blank', 'repair']), out / 'val_scope.parquet')
    atomic_parquet(val_text.filter(pl.col('repair')).select(KEYS + ['a', 'b']), out / 'val_pairs.parquet')
    info = pl.scan_parquet(work / 'norm/train_source1.parquet').filter(
        pl.col('entity_id').is_in(all_val.implode())).select('entity_id', 'country', 'name_core').collect()
    if info.height != all_val.len():
        raise ValueError('Missing validation entity metadata')
    atomic_parquet(info, out / 'val_entities.parquet')
    train_text = texts(train, attrs(work, 'train', train))
    summary = {'validation_entities': info.height, 'validation_repair_pairs': int(val_text['repair'].sum()), 'folds': {}}
    for fold in (0, 1):
        fit = train_text.filter(pl.col('fold') != fold)
        positive = fit.filter(pl.col('y') == 1)
        hard = fit.filter((pl.col('y') == 0) & ((pl.col('p') >= .2) | pl.col('blank')))
        easy = fit.filter((pl.col('y') == 0) & (pl.col('p') < .2) & ~pl.col('blank'))
        # Fixed 50/35/15 budget; keep all rows when a stratum is smaller than its budget.
        strata = [(positive, .5, 1.5), (hard, .35, 1.5), (easy, .15, 1.)]
        selected = pl.concat([frame.sample(n=min(frame.height, int(args.max_pairs * share)),
            seed=42 + fold, shuffle=True).with_columns(pl.lit(weight).alias('weight'))
            for frame, share, weight in strata]).sample(fraction=1., seed=42 + fold, shuffle=True)
        if selected['y'].n_unique() != 2 or selected.height < 1000:
            raise ValueError(f'Fold {fold} has insufficient labeled data')
        atomic_parquet(selected.select(KEYS + ['a', 'b', 'y', 'weight']), out / f'fit_fold{fold}.parquet')
        summary['folds'][str(fold)] = dict(pairs=selected.height, positives=int(selected['y'].sum()),
            hard_negatives=min(hard.height, int(args.max_pairs * .35)))
        print(f'PREPARED fold {fold}: {summary["folds"][str(fold)]}', flush=True)
    atomic_json(out / 'PREPARE_COMPLETE.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
