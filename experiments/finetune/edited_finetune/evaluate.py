"""Calibrate on business groups, evaluate entity F0.5, export only on acceptance."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from common import (KEYS, atomic_json, atomic_parquet, code_digest, digest,
                    owners, vector, verify_pairs)


def collect_scores(out, split):
    expected = pl.read_parquet(out / f'{split}_pairs.parquet', columns=KEYS)
    parts = []
    for fold in (0, 1):
        folder = out / f'{split}_scores_fold{fold}'
        if not (folder / 'COMPLETE.json').exists():
            raise ValueError(f'Incomplete inference: {folder}')
        files = sorted(folder.glob('*.parquet'))
        if not files:
            raise ValueError('No repair scoring shards')
        frame = pl.concat([pl.read_parquet(path) for path in files])
        verify_pairs(frame)
        if frame.height != expected.height or frame.join(expected, on=KEYS, how='anti').height:
            raise ValueError('Repair score coverage mismatch')
        if not frame['ft_logit'].is_finite().all():
            raise ValueError('Nonfinite fine-tuned scores')
        parts.append(frame)
    return pl.concat(parts).group_by(KEYS).agg(pl.col('ft_logit').mean())


def matrix(frame, fine_tuned):
    p = np.clip(frame['p2'].to_numpy(), 1e-6, 1 - 1e-6)
    blank = frame['blank'].to_numpy().astype(float)
    values = [np.log(p / (1 - p)), blank]
    if fine_tuned:
        score = np.clip(frame['ft_logit'].to_numpy(), -20, 20) / 4
        values += [score, score * blank]
    result = np.column_stack(values)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite score-fusion features')
    return result


def fused(frame, parameters):
    routed = frame.filter(pl.col('repair'))
    x = matrix(routed, parameters['fine_tuned'])
    z = np.clip((x * np.asarray(parameters['coef'])).sum(axis=1) + parameters['intercept'], -40, 40)
    update = routed.select(KEYS).with_columns(pl.Series('_new', 1 / (1 + np.exp(-z))))
    return frame.join(update, on=KEYS, how='left', validate='1:1').with_columns(
        pl.col('_new').fill_null(pl.col('p2')).alias('p2')).drop('_new')


def select(frame, parameters, exact_best_k):
    kept = owners(frame)
    q = np.interp(kept['p2'].to_numpy(), parameters['x'], parameters['y'])
    d = kept.select(KEYS + ['p2']).with_columns(pl.Series('q', q)).filter(pl.col('q') >= .01)
    d = d.sort(['s1', 'q', 'p2', 'cand'], descending=[False, True, True, False]).with_columns(
        pl.int_range(pl.len()).over('s1').alias('_rank')).filter(pl.col('_rank') < 20)
    if d.is_empty():
        return d.select(KEYS)
    d = d.with_columns((pl.col('s1').rank('dense') - 1).cast(pl.Int64).alias('_entity'))
    probabilities = np.zeros((int(d['_entity'].max()) + 1, 20))
    probabilities[d['_entity'].to_numpy(), d['_rank'].to_numpy()] = d['q'].to_numpy()
    k = exact_best_k(probabilities)
    return d.filter(pl.col('_rank') < pl.Series(k[d['_entity'].to_numpy()])).select(KEYS)


def compare(a, b, info):
    delta = b - a
    countries = np.asarray(info['country'].to_list())
    groups = info.select('country', 'name_core', 'entity_id').iter_rows()
    buckets = {}
    for index, (country, name, entity) in enumerate(groups):
        buckets.setdefault((country, name or entity), []).append(index)
    rng = np.random.default_rng(42)
    draws, sizes = np.zeros(1000), np.zeros(1000)
    for country in sorted(set(countries)):
        indices = [idx for group, idx in buckets.items() if group[0] == country]
        sums = np.array([delta[idx].sum() for idx in indices])
        counts = np.array([len(idx) for idx in indices])
        for offset in range(0, 1000, 20):
            sample = rng.integers(len(indices), size=(20, len(indices)))
            draws[offset:offset + 20] += sums[sample].sum(1)
            sizes[offset:offset + 20] += counts[sample].sum(1)
    return dict(reference=float(a.mean()), variant=float(b.mean()), delta=float(delta.mean()),
        business_group_bootstrap_ci95=np.quantile(draws / sizes, [.025, .975]).tolist(),
        country={c: dict(reference=float(a[countries == c].mean()),
                        variant=float(b[countries == c].mean()), delta=float(delta[countries == c].mean()))
                 for c in sorted(set(countries))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['validation', 'test'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--code', required=True)
    args = parser.parse_args()
    out, code = Path(args.out).resolve(), Path(args.code).resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    if manifest['code_sha256'] != code_digest():
        raise ValueError('Prepared input/code mismatch')
    work = Path(manifest['work'])
    if digest(work / 'lgb_s2.json') != manifest['baseline_meta_sha256']:
        raise ValueError('Baseline metadata changed')
    os.environ['BER_WORK_DIR'] = str(work)
    os.environ['BER_OUT_DIR'] = str(out / 'output')
    sys.path.insert(0, str(code / 'src' if (code / 'src').is_dir() else code))
    from decision import exact_best_k
    from decide import write_lists
    if args.mode == 'test':
        report = json.loads((out / 'comparison.json').read_text())
        if not report['eligible_for_test_prediction']:
            raise ValueError('No accepted fine-tuning improvement')
        target = out / 'output/matching_results_finetuned.tsv'
        if target.exists():
            print(f'TSV already exists: {target}', flush=True)
            return
        scope = pl.read_parquet(out / 'test_scope.parquet').join(collect_scores(out, 'test'), on=KEYS, how='left', validate='1:1')
        if scope.filter(pl.col('repair'))['ft_logit'].null_count():
            raise ValueError('Missing repair scores')
        parameters = json.loads((out / 'fine_tuned_decision.json').read_text())
        scored = fused(scope, parameters['fusion'])
        final = select(scored, parameters['calibration'], exact_best_k)
        if final['cand'].n_unique() != final.height:
            raise ValueError('One-owner-per-candidate constraint violated')
        ids = pl.read_parquet(work / 'norm/test_source1.parquet', columns=['entity_id'])['entity_id'].to_list()
        write_lists(ids, scope.select(KEYS), out / 'output/candidate_pairs.tsv', 'candidate_entity_ids')
        write_lists(ids, final, target.with_suffix('.tsv.partial'), 'matched_entity_ids')
        os.replace(target.with_suffix('.tsv.partial'), target)
        atomic_json(out / 'TSV_CREATED.json', dict(file=str(target), sha256=digest(target)))
        print('FINETUNED_TSV=' + str(target), flush=True)
        return
    if (out / 'comparison.json').exists():
        print((out / 'comparison.json').read_text(), flush=True)
        return
    scope = pl.read_parquet(out / 'val_scope.parquet').join(collect_scores(out, 'val'), on=KEYS, how='left', validate='1:1')
    if scope.filter(pl.col('repair'))['ft_logit'].null_count():
        raise ValueError('Missing validation repair scores')
    info = pl.read_parquet(out / 'val_entities.parquet').sort('entity_id')
    # Whole country + normalized-name groups stay together. Fixed 40/60 split.
    import hashlib
    role = [int.from_bytes(hashlib.sha256((country + '|' + (name or entity)).encode()).digest()[:8], 'little') % 100 < 40
            for entity, country, name in info.select('entity_id', 'country', 'name_core').fill_null('').iter_rows()]
    cal_info, hold_info = info.filter(pl.Series(role)), info.filter(~pl.Series(role))
    if min(cal_info.height, hold_info.height) < 5000:
        raise ValueError('Calibration/evaluation partitions too small')
    cal_ids, hold_ids = cal_info['entity_id'], hold_info['entity_id'].to_list()
    for country in info['country'].unique():
        if min(cal_info.filter(pl.col('country') == country).height,
               hold_info.filter(pl.col('country') == country).height) < 500:
            raise ValueError('Country partition too small')
    atomic_json(out / 'split.json', dict(cal=cal_ids.to_list(), hold=hold_ids))
    truth = pl.read_parquet(work / 'raw/train_ground_truth_long.parquet').rename(
        {'source1_entity_id': 's1', 'match': 'cand'})
    truth = truth.filter(pl.col('s1').is_in(info['entity_id'].implode()))
    calibration = scope.filter(pl.col('repair') & pl.col('s1').is_in(cal_ids.implode()))
    if calibration.height < 500 or calibration['y'].n_unique() != 2:
        raise ValueError('Repair calibration needs sufficient pairs and both labels')
    results = {}
    for name, fine_tuned in [('control', False), ('fine_tuned', True)]:
        model = LogisticRegression(C=1., max_iter=1000, random_state=42)
        model.fit(matrix(calibration, fine_tuned), calibration['y'].to_numpy())
        fusion = dict(fine_tuned=fine_tuned, coef=model.coef_[0].tolist(), intercept=float(model.intercept_[0]))
        scored = fused(scope, fusion)
        cal = owners(scored).filter(pl.col('s1').is_in(cal_ids.implode()))
        if cal['y'].n_unique() != 2:
            raise ValueError('Ownership calibration lacks classes')
        iso = IsotonicRegression(out_of_bounds='clip').fit(cal['p2'].to_numpy(), cal['y'].to_numpy())
        parameters = dict(x=iso.X_thresholds_.tolist(), y=iso.y_thresholds_.tolist())
        atomic_json(out / f'{name}_decision.json', dict(fusion=fusion, calibration=parameters))
        print(f'ENTITY_SELECTION {name}', flush=True)
        pred = select(scored, parameters, exact_best_k)
        atomic_parquet(pred, out / f'val_pred_{name}.parquet')
        results[name] = vector(pred, truth, hold_ids)
        print(f'{name}: evaluation entity F0.5={results[name].mean():.9f}', flush=True)
    baseline = vector(pl.read_parquet(work / 'val_pred_s2.parquet'), truth, hold_ids)
    vs_control = compare(results['control'], results['fine_tuned'], hold_info)
    vs_baseline = compare(baseline, results['fine_tuned'], hold_info)
    passed = all(c['delta'] >= .0002 and c['business_group_bootstrap_ci95'][0] > 0 and
                 all(country['delta'] >= 0 for country in c['country'].values()) for c in (vs_control, vs_baseline))
    report = dict(code_sha256=code_digest(), eligible_for_test_prediction=bool(passed),
        calibration_entities=cal_info.height, evaluation_entities=hold_info.height,
        repair_pairs=int(scope['repair'].sum()), versus_control=vs_control, versus_saved_baseline=vs_baseline,
        limitations=['Reused validation: exploratory evidence, not a new independent holdout.',
            'Original checkpoint/upstream fitting previously inspected validation; extra fine-tuning uses no validation labels.',
            'No French ground truth. No leaderboard guarantee.'])
    atomic_json(out / 'comparison.json', report)
    print(json.dumps(report, indent=2), flush=True)
    print('ACCEPTED_FOR_TEST_SCORING' if passed else 'NO_ACCEPTED_IMPROVEMENT: retain verified baseline.', flush=True)


if __name__ == '__main__':
    main()
