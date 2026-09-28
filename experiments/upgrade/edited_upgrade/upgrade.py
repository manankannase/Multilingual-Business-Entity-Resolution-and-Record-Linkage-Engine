"""Compare mixed competition-dropout training with the saved Edited_version baseline."""
import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['fit', 'predict'])
    ap.add_argument('--code', required=True)
    ap.add_argument('--base-work', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    base, out = Path(args.base_work).resolve(), Path(args.out).resolve()
    if out == base or base in out.parents:
        ap.error('Use an output directory outside the baseline work directory')
    code = Path(args.code).resolve()
    if (code / 'src/stage2.py').exists():
        code = code / 'src'
    if not (code / 'stage2.py').exists():
        ap.error('stage2.py not found in code or code/src')
    os.environ['BER_WORK_DIR'] = str(base)
    os.environ['BER_OUT_DIR'] = str(out / 'output')
    sys.path.insert(0, str(code))
    import stage2 as s2
    import decision
    from decide import one_to_one, write_lists
    from data import load_ground_truth
    from prep import scan_norm

    out.mkdir(parents=True, exist_ok=True)
    # flock prevents duplicate runs from writing the same models/output.
    import fcntl
    lock = open(out / '.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    decision.DEC_PATH = out / 'decision.pkl'
    meta = json.loads((base / 'lgb_s2.json').read_text())
    cols, tags = meta['cols'], meta['tags']
    if tags != ['xenc']:
        raise ValueError('This experiment expects the verified XENC-only Stage-2 baseline')
    code_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def context(scored, split, name):
        d = s2.build_ctx(scored, split, name, tags)
        if d['xenc'].null_count():
            raise ValueError('Missing cached XENC scores: input shortlist must stay unchanged')
        return d.join(pl.read_parquet(base / 'feat' / f'{name}.parquet',
                      columns=['s1', 'cand'] + meta['stage1_cols']),
                      on=['s1', 'cand'], how='inner', validate='1:1')

    def dropped(scored, frac, seed):
        ids = scored['s1'].unique().sort().to_numpy()
        removed = np.random.default_rng(seed).choice(ids, int(len(ids) * frac), replace=False)
        return scored.filter(~pl.col('s1').is_in(pl.Series(removed).implode()))

    def predict_scores(models, d):
        return d.with_columns(pl.Series('p2', s2._predict(models, s2._X(d, cols))))

    def apply(d, path):
        decision.DEC_PATH = path
        return decision.apply(d)

    if args.mode == 'predict':
        report = json.loads((out / 'comparison.json').read_text())
        if report['code_sha256'] != code_digest:
            raise ValueError('Code changed since fit')
        if not report['eligible_for_test_prediction']:
            raise SystemExit('Enhancement did not pass comparison; retain the 0.981 baseline')
        target = out / 'output'
        if (target / 'matching_results.tsv').exists():
            raise FileExistsError('Output already exists; refusing overwrite')
        models = [lgb.Booster(model_file=str(out / f'model_{i}.txt')) for i in range(3)]
        scored = pl.read_parquet(base / 'test_scored.parquet')
        ctx = s2.build_ctx(scored, 'test', 'test', tags)
        if ctx['xenc'].null_count():
            raise ValueError('Missing test XENC scores')
        ctx = ctx.select(['s1', 'cand'] + [c for c in cols if c not in meta['stage1_cols']])
        parts = []
        for file in sorted((base / 'feat/test_shards').glob('*.parquet')):
            sh = pl.read_parquet(file, columns=['s1', 'cand'] + meta['stage1_cols'])
            sh = sh.join(ctx, on=['s1', 'cand'], how='inner', validate='1:1')
            if sh.height:
                scored_sh = predict_scores(models, sh)
                parts.append(scored_sh.select(['s1', 'cand', 'p2'] +
                             [c for c in decision.needed_cols() if c in sh.columns]))
        te = pl.concat(parts)
        if te.height != ctx.height:
            raise ValueError('Test shard coverage mismatch')
        te.write_parquet(out / 'test_scored_upgrade.parquet')
        final = apply(te, out / 'decision.pkl')
        ids = scan_norm('test', 1, ['entity_id'])['entity_id'].to_list()
        write_lists(ids, final, target / 'matching_results.tsv', 'matched_entity_ids')
        # Report the actual shortlist scored by the final model.
        write_lists(ids, te.select('s1', 'cand'), target / 'candidate_pairs.tsv', 'candidate_entity_ids')
        print('GENERATED_MATCHING_TSV=' + str(target / 'matching_results.tsv'), flush=True)
        return

    if (out / 'comparison.json').exists() or (out / 'model_0.txt').exists():
        raise FileExistsError('Fit output already exists; use a fresh output directory')
    tr_scores = pl.read_parquet(base / 's2_train_p.parquet').drop('fold')
    va_scores = pl.read_parquet(base / 's2_val_p.parquet')
    tr0 = context(tr_scores, 'train', 'train')
    # Recompute the competition graph while preserving original pair evidence,
    # retrieval ranks and OOF probabilities. This is a stress augmentation, not
    # a reconstruction of the hidden test distribution.
    tr1 = context(dropped(tr_scores, .10, 53), 'train', 'train')
    tr2 = context(dropped(tr_scores, .20, 54), 'train', 'train')
    tr = pl.concat([tr0.select(['y'] + cols).with_columns(pl.lit(.5).alias('_w')),
                    tr1.select(['y'] + cols).with_columns(pl.lit(.25 / .9).alias('_w')),
                    tr2.select(['y'] + cols).with_columns(pl.lit(.25 / .8).alias('_w'))],
                    how='vertical_relaxed')
    del tr0, tr1, tr2, tr_scores
    va = context(va_scores, 'train', 'val')
    all_ids = s2.s1_subset('train')['val'].to_list()
    if not set(va['s1']).issubset(set(all_ids)):
        raise ValueError('Sampling settings do not match cached validation; check BER_N_VAL_S1')
    truth = load_ground_truth().rename({'source1_entity_id': 's1', 'match': 'cand'})
    truth = truth.filter(pl.col('s1').is_in(pl.Series(all_ids).implode()))
    info = scan_norm('train', 1, ['entity_id', 'country', 'state', 'name_core'])
    info = info.filter(pl.col('entity_id').is_in(pl.Series(all_ids).implode()))
    country_states = dict(info.group_by('country').agg(pl.col('state').n_unique()).iter_rows())
    # Small regional samples may have too few states for two calibration folds
    # plus a holdout. Fall back to whole normalized-name groups in that country.
    group_of = {s: (c, ('state:' + str(st)) if country_states[c] >= 4 else
                        ('name:' + str(name or s))) for s,c,st,name in info.iter_rows()}
    groups = sorted(set(group_of.values()))
    rng = np.random.default_rng(112)
    cal_groups = set()
    fold_group = {}
    for country in sorted(country_states):
        states = [st for c,st in groups if c == country]
        rng.shuffle(states)
        if len(states) < 4:
            raise ValueError('Need at least four validation groups per country')
        for i, st in enumerate(states[:max(2, len(states)//2)]):
            cal_groups.add((country, st)); fold_group[(country, st)] = i % 2
    cal_ids = [s for s in all_ids if group_of[s] in cal_groups]
    hold_ids = [s for s in all_ids if group_of[s] not in cal_groups]
    vc = va.filter(pl.col('s1').is_in(pl.Series(cal_ids).implode()))
    if set(vc['y'].unique()) != {0, 1}:
        raise ValueError('Calibration group needs both labels')
    print(f'fit rows {tr.height:,}; calibration entities {len(cal_ids):,}; holdout entities {len(hold_ids):,}', flush=True)
    dtr = lgb.Dataset(s2._X(tr, cols), tr['y'].to_numpy(), weight=tr['_w'].to_numpy(), feature_name=cols)
    dva = lgb.Dataset(s2._X(vc, cols), vc['y'].to_numpy(), reference=dtr)
    models = []
    for i in range(3):
        params = dict(s2.S2_PARAMS, seed=42+i, bagging_seed=42+i, feature_fraction_seed=42+i)
        m = lgb.train(params, dtr, 4000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
        m.save_model(str(out / f'model_{i}.txt')); models.append(m)
    del tr, dtr, dva
    new_va = predict_scores(models, va)
    cal = new_va.filter(pl.col('s1').is_in(pl.Series(cal_ids).implode()))
    kept = one_to_one(cal, 'p2')
    folds = {s: fold_group[group_of[s]] for s in cal_ids}
    kfold = np.array([folds[s] for s in kept['s1']])
    q = np.zeros(kept.height)
    for f in (0, 1):
        fit = kfold != f; test = kfold == f
        if not fit.any() or not test.any():
            raise ValueError('Empty calibration fold')
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(kept['p2'].to_numpy()[fit], kept['y'].to_numpy()[fit])
        q[test] = iso.predict(kept['p2'].to_numpy()[test])
    from decide import evaluate
    grid = np.round(np.arange(.3, .96, .02), 2)
    t, ft = max(((float(t), evaluate(kept.filter(pl.col('p2') >= t).select('s1','cand'), truth, cal_ids))
                 for t in grid), key=lambda x: x[1])
    fe = evaluate(decision.select_exact(kept, q), truth, cal_ids)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(kept['p2'].to_numpy(), kept['y'].to_numpy())
    dec = {'rule': 'exact' if fe >= ft else 'threshold', 'score': 'p2', 'calib': iso, 'thr': t}
    with open(out / 'decision.pkl', 'wb') as f:
        pickle.dump(dec, f)
    new_pred = apply(new_va, out / 'decision.pkl')
    old_pred = pl.read_parquet(base / 'val_pred_s2.parquet')

    def vector(pred, ids):
        pp, tt = {}, {}
        for s, c in pred.select('s1','cand').iter_rows():
            pp.setdefault(s, set()).add(c)
        for s, c in truth.select('s1','cand').iter_rows():
            tt.setdefault(s, set())
            if c is not None: tt[s].add(c)
        from metric import entity_f05
        return np.array([entity_f05(pp.get(s,set()), tt.get(s,set())) for s in ids])

    old, new = vector(old_pred, hold_ids), vector(new_pred, hold_ids)
    delta = new-old
    # Paired confidence interval clusters by country/state, rather than treating
    # correlated records as independent observations.
    state_of = group_of
    clusters = {}
    for s,d in zip(hold_ids,delta): clusters.setdefault(state_of[s],[]).append(d)
    sums = np.array([sum(a) for a in clusters.values()]); ns = np.array([len(a) for a in clusters.values()])
    picks = np.random.default_rng(113).integers(len(sums), size=(2000,len(sums)))
    ci = np.quantile(sums[picks].sum(1)/ns[picks].sum(1), [.025,.975]).tolist()
    by_country = {}
    for c in info['country'].unique().sort():
        mask = np.array([state_of[s][0] == c for s in hold_ids])
        by_country[c] = {'baseline': float(old[mask].mean()), 'upgrade': float(new[mask].mean()),
                         'delta': float(delta[mask].mean())}
    # Stress comparison: remove owners from validation, recompute context, then
    # evaluate old/new models under exactly the same constructed input graph.
    removed = set(np.random.default_rng(115).choice(sorted(all_ids), int(len(all_ids)*.20), replace=False))
    stress_scores = va_scores.filter(~pl.col('s1').is_in(pl.Series(sorted(removed)).implode()))
    stress = context(stress_scores, 'train', 'val')
    stress_ids = [s for s in hold_ids if s not in removed]
    old_models = [lgb.Booster(model_file=str(base / f'lgb_s2_{i}.txt')) for i in range(meta['n_models'])]
    old_stress = apply(predict_scores(old_models, stress), base / 'decision.pkl')
    new_stress = apply(predict_scores(models, stress), out / 'decision.pkl')
    oscore, nscore = vector(old_stress, stress_ids), vector(new_stress, stress_ids)
    eligible = bool(delta.mean() >= .0002 and ci[0] > 0 and
                    all(x['delta'] >= 0 for x in by_country.values()) and nscore.mean() >= oscore.mean())
    report = {'baseline_public_score': .981, 'code_sha256': code_digest,
              'holdout_entities': len(hold_ids), 'calibration_entities': len(cal_ids),
              'holdout_baseline_f05': float(old.mean()), 'holdout_upgrade_f05': float(new.mean()),
              'delta': float(delta.mean()), 'group_cluster_delta_ci95': ci, 'by_country': by_country,
              'stress_baseline_f05': float(oscore.mean()), 'stress_upgrade_f05': float(nscore.mean()),
              'stress_entities': len(stress_ids), 'decision': dec['rule'],
              'eligible_for_test_prediction': eligible,
              'split_group_by_country': {c: 'state' if n >= 4 else 'normalized_name' for c,n in country_states.items()},
              'limitation': 'Baseline previously used all validation; new holdout is unused for new early stopping/calibration. Dropped-owner context is a synthetic stress condition; raw retrieval features remain fixed. No French labels.'}
    (out / 'comparison.json').write_text(json.dumps(report,indent=2))
    new_va.select('s1','cand','p','p2','y').write_parquet(out / 'val_scored_upgrade.parquet')
    new_pred.write_parquet(out / 'val_pred_upgrade.parquet')
    print(json.dumps(report,indent=2), flush=True)


if __name__ == '__main__':
    main()
