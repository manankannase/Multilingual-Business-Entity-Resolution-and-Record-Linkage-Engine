"""Synthetic integration checks; no GPU or transformer dependency."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from common import KEYS, atomic_json, atomic_parquet, attrs, owners, repair_scope, texts, vector
from evaluate import collect_scores, fused, matrix


def run(*args, success=True):
    result = subprocess.run([sys.executable, *map(str, args)], text=True, capture_output=True)
    if (result.returncode == 0) != success:
        raise AssertionError(result.stdout + result.stderr)
    return result


def main():
    here = Path(__file__).resolve().parent
    code = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1
            else here.parents[2] / 'src/business_entity_resolution')
    with tempfile.TemporaryDirectory(prefix='edited-ft-test-') as tmp:
        work, out = Path(tmp) / 'work', Path(tmp) / 'run'
        for folder in ['raw', 'norm', 'cand', 'models/xenc_fold0', 'models/xenc_fold1']:
            (work / folder).mkdir(parents=True, exist_ok=True)
        for fold in (0, 1):
            atomic_json(work / f'models/xenc_fold{fold}/config.json', {'num_labels': 1})
            # Preparation only checks checkpoint discovery, not weights loading.
            (work / f'models/xenc_fold{fold}/pytorch_model.bin').write_bytes(b'fixture')
        atomic_json(work / 'lgb_s2.json', {'tags': ['xenc']})
        ntrain, nval = 8000, 14000
        sids = [f'T{i}' for i in range(ntrain)] + [f'V{i}' for i in range(nval)]
        cids = [f'C{i}' for i in range(ntrain)] + [f'P{i}' for i in range(nval)] + [f'N{i}' for i in range(nval)]
        def attributes(ids):
            return pl.DataFrame({'entity_id': ids, 'name_clean': ['business ' + s for s in ids],
                'name_core': ['business ' + s for s in ids], 'addr_clean': ['123 road'] * len(ids),
                'business_address': ['123 road'] * len(ids), 'state': ['x'] * len(ids),
                'country': ['US' if i % 2 else 'India' for i in range(len(ids))]})
        attributes(sids).write_parquet(work / 'norm/train_source1.parquet')
        attributes(cids).write_parquet(work / 'norm/train_source2.parquet')
        attributes([]).write_parquet(work / 'norm/train_source3.parquet')
        train = pl.DataFrame({'s1': [f'T{i}' for i in range(ntrain)], 'cand': [f'C{i}' for i in range(ntrain)],
            'p': [.8 if i % 4 < 2 else .3 if i % 4 == 2 else .1 for i in range(ntrain)],
            'fold': [i % 2 for i in range(ntrain)], 'y': [int(i % 4 < 2) for i in range(ntrain)]})
        train.write_parquet(work / 's2_train_p.parquet')
        val = pl.DataFrame({'s1': [f'V{i}' for i in range(nval)] * 2,
            'cand': [f'P{i}' for i in range(nval)] + [f'N{i}' for i in range(nval)],
            'p': [.8] * nval + [.1] * nval, 'p2': [.8] * nval + [.1] * nval,
            'y': [1] * nval + [0] * nval})
        val.write_parquet(work / 'val_scored_s2.parquet')
        val.select('s1').write_parquet(work / 'cand/val.parquet')
        pred = val.filter(pl.col('y') == 1).select(KEYS)
        pred.write_parquet(work / 'val_pred_s2.parquet')
        pred.rename({'s1': 'source1_entity_id', 'cand': 'match'}).write_parquet(work / 'raw/train_ground_truth_long.parquet')
        for mode in ['preflight', 'prepare', 'prepare']:
            run(here / 'prepare.py', mode, '--work', work, '--out', out, '--max-pairs', '4000')
        for fold in (0, 1):
            fit = pl.read_parquet(out / f'fit_fold{fold}.parquet')
            original = fit.join(train.select(KEYS + ['fold']), on=KEYS)
            assert (original['fold'] != fold).all()
            assert set(fit['s1']).isdisjoint(val['s1'])
            assert set(fit['cand']).isdisjoint(val['cand'])
            assert fit['y'].n_unique() == 2
        expected = pl.read_parquet(out / 'val_pairs.parquet').select(KEYS)
        for fold in (0, 1):
            folder = out / f'val_scores_fold{fold}'
            folder.mkdir()
            score = expected.with_columns(pl.when(pl.col('cand').str.starts_with('P')).then(5.).otherwise(-5.).alias('ft_logit'))
            atomic_parquet(score, folder / '000000000000.parquet')
            atomic_json(folder / 'COMPLETE.json', {'pairs': expected.height})
        combined = collect_scores(out, 'val')
        assert combined.height == expected.height
        scope = pl.read_parquet(out / 'val_scope.parquet').join(combined, on=KEYS)
        assert matrix(scope, True).shape == (scope.height, 4)
        updated = fused(scope, {'fine_tuned': True, 'coef': [0., 0., 4., 0.], 'intercept': 0.})
        assert updated.filter(pl.col('y') == 1)['p2'].min() > .99
        assert np.allclose(vector(pred, pred, [f'V{i}' for i in range(nval)]), 1.)
        assert owners(updated)['cand'].n_unique() == owners(updated).height
        # Corrupt/duplicate score shards must fail before evaluation.
        atomic_parquet(combined.head(1), out / 'val_scores_fold0/duplicate.parquet')
        try:
            collect_scores(out, 'val')
        except ValueError:
            pass
        else:
            raise AssertionError('Duplicate inference scores accepted')
        (out / 'val_scores_fold0/duplicate.parquet').unlink()
        # Full CPU evaluation uses original expected-F selector; baseline is perfect,
        # so no fine-tuning candidate can satisfy promotion thresholds.
        run(here / 'evaluate.py', 'validation', '--out', out, '--code', code)
        report = json.loads((out / 'comparison.json').read_text())
        assert not report['eligible_for_test_prediction']
        run(here / 'prepare.py', 'test', '--work', work, '--out', out, '--max-pairs', '4000', success=False)
        assert not (out / 'test_pairs.parquet').exists()
        # Input drift must prevent preparation/resume.
        atomic_json(work / 'lgb_s2.json', {'changed': True})
        run(here / 'prepare.py', 'prepare', '--work', work, '--out', out, '--max-pairs', '4000', success=False)
    print('PASS: preparation/resume, fold isolation, score coverage, entity evaluation, rejection gate, input drift')


if __name__ == '__main__':
    main()
