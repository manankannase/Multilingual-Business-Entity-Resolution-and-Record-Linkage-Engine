"""Matched Stage-2 experiment using cached upstream models; no baseline writes."""
import argparse
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

# Avoid collision with the original project's features.py.
import importlib.util
spec = importlib.util.spec_from_file_location('micro_features', Path(__file__).with_name('features.py'))
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)
KEYS = ['s1', 'cand']
CANDIDATES = ['01_expected_f', '02_threshold', '03_missing_calibration',
              '04_model_ensemble', '05_missing_only_update']


def owners(d):
    return d.sort(['p2','s1','cand'], descending=[True,False,False]).unique('cand', keep='first')


def vector(pred, truth, ids):
    pp, tt = {}, {}
    for s,c in pred.select(KEYS).iter_rows():
        pp.setdefault(s,set()).add(c)
    for s,c in truth.select(KEYS).iter_rows():
        if c is not None:
            tt.setdefault(s,set()).add(c)
    vals = []
    for s in ids:
        p, t = pp.get(s,set()), tt.get(s,set())
        vals.append(1.25*len(p&t)/(.25*len(t)+len(p)) if t or p else 1.)
    return np.array(vals)


def partition(info):
    """Balance counts by state; use whole normalized names if states cannot balance."""
    if info['entity_id'].n_unique() != info.height:
        raise ValueError('Duplicate validation Source-1 IDs')
    result, group_of = {}, {}
    for country in info['country'].unique().sort():
        sub = info.filter(pl.col('country') == country)
        state_counts = dict(sub.with_columns(pl.col('state').fill_null('')).group_by('state').len().sort('state').iter_rows())
        states, counts = list(state_counts), np.array(list(state_counts.values()))
        n = sub.height
        targets, minimum = np.array([.2,.3,.5]), np.array([.1,.1,.25])*n
        choices = []
        # Existing cached split has five/six states; enumeration is bounded.
        if 5 <= len(states) <= 10:
            for assignment in itertools.product(range(3),repeat=len(states)):
                a = np.array(assignment)
                if (a==0).sum()<1 or (a==1).sum()<2 or (a==2).sum()<2:
                    continue
                sizes = np.array([counts[a==i].sum() for i in range(3)])
                if (sizes < minimum).any():
                    continue
                # Prevent one tiny state from constituting a calibration fold.
                cost = float(np.square(sizes/n-targets).sum())
                choices.append((cost,assignment))
        if choices:
            assignment = min(choices)[1]
            roles = {st: ['stop','cal','hold'][i] for st,i in zip(states,assignment)}
            for s,st in sub.select('entity_id','state').iter_rows():
                result[s] = roles[st or '']
                group_of[s] = (country,'state:'+(st or ''))
        else:
            # All claimants with the same normalized name stay in one partition.
            names = {s: ('name:'+name) if name else ('id:'+s)
                     for s,name in sub.select('entity_id','name_core').fill_null('').iter_rows()}
            sizes = {}
            for name in names.values():
                sizes[name] = sizes.get(name,0)+1
            if len(sizes)<10:
                raise ValueError(f'{country}: insufficient independent business-name groups')
            totals, roles = np.zeros(3,dtype=int), {}
            for name,size in sorted(sizes.items(),key=lambda x:(-x[1],hashlib.sha256(x[0].encode()).hexdigest())):
                role = int(np.argmax((targets*n-totals)/(targets*n)))
                roles[name] = ['stop','cal','hold'][role]
                totals[role] += size
            if (totals < minimum).any():
                raise ValueError(f'{country}: unable to balance independent name groups')
            for s,name in names.items():
                result[s], group_of[s] = roles[name], (country,name)
        actual = {role:sum(result[s]==role for s in sub['entity_id']) for role in ['stop','cal','hold']}
        if any(actual[r]<minimum[i] for i,r in enumerate(['stop','cal','hold'])):
            raise ValueError(f'{country}: undersized partition {actual}')
    return result, group_of


def calibration_folds(info, cal_ids):
    """Two balanced folds of whole names within calibration, separately by country."""
    sub = info.filter(pl.col('entity_id').is_in(pl.Series(cal_ids).implode()))
    result = {}
    for country in sub['country'].unique().sort():
        data = sub.filter(pl.col('country')==country)
        names = {s: name or ('id:'+s) for s,name in data.select('entity_id','name_core').fill_null('').iter_rows()}
        counts = {}
        for name in names.values():
            counts[name] = counts.get(name,0)+1
        sizes, fold_of = [0,0], {}
        for name,n in sorted(counts.items(),key=lambda x:(-x[1],hashlib.sha256(x[0].encode()).hexdigest())):
            fold = int(sizes[1]<sizes[0])
            fold_of[name] = fold
            sizes[fold] += n
        if min(sizes)<.2*sum(sizes):
            raise ValueError(f'{country}: calibration name groups cannot form balanced folds')
        result.update({s:fold_of[name] for s,name in names.items()})
    return result


def five_scores(variant, baseline):
    joined = variant.join(baseline.select(KEYS+['p2']).rename({'p2':'_baseline'}),on=KEYS,how='left',validate='1:1')
    if joined['_baseline'].null_count():
        raise ValueError('Baseline/variant score coverage mismatch')
    plain = joined.drop('_baseline')
    return {CANDIDATES[0]:plain,CANDIDATES[1]:plain,CANDIDATES[2]:plain,
        CANDIDATES[3]:joined.with_columns((.75*pl.col('p2')+.25*pl.col('_baseline')).alias('p2')).drop('_baseline'),
        CANDIDATES[4]:joined.with_columns(pl.when(pl.col('ma_cand_blank')==1).then(pl.col('p2'))
            .otherwise(pl.col('_baseline')).alias('p2')).drop('_baseline')}


def comparison(a, b, ids, groups):
    delta = b-a
    country = {}
    for c in sorted({groups[s][0] for s in ids}):
        mask = np.array([groups[s][0] == c for s in ids])
        country[c] = {'reference': float(a[mask].mean()), 'variant': float(b[mask].mean()),
                      'delta': float(delta[mask].mean())}
    # Paired, country-stratified state bootstrap. Few states => approximate interval.
    rng, draws = np.random.default_rng(928), np.zeros(2000)
    sizes = np.zeros(2000)
    for c in country:
        buckets = {}
        for s,d in zip(ids,delta):
            if groups[s][0] == c:
                buckets.setdefault(groups[s], []).append(d)
        sums = np.array([sum(v) for v in buckets.values()])
        ns = np.array([len(v) for v in buckets.values()])
        for start in range(0,2000,100):
            picks = rng.integers(len(sums), size=(100,len(sums)))
            draws[start:start+100] += sums[picks].sum(1)
            sizes[start:start+100] += ns[picks].sum(1)
    return {'reference': float(a.mean()), 'variant': float(b.mean()), 'delta': float(delta.mean()),
            'group_bootstrap_ci95': np.quantile(draws/sizes,[.025,.975]).tolist(), 'country':country}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight','fit','predict'])
    parser.add_argument('--code', required=True)
    parser.add_argument('--base-work', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()
    base, out, code = (Path(p).resolve() for p in (args.base_work,args.out,args.code))
    if out == base or base in out.parents or out in base.parents:
        raise ValueError('Output must be separate from baseline work')
    if (code/'src/stage2.py').exists():
        code = code/'src'
    if not (code/'stage2.py').exists() or not (base/'lgb_s2.json').exists():
        raise FileNotFoundError('Missing source code or Stage-2 baseline metadata')
    if not (base/'raw/train_ground_truth_long.parquet').exists():
        raise FileNotFoundError('Expected cached ground truth; refusing to create files in baseline work')
    if not 1 <= args.seeds <= 3:
        raise ValueError('seeds must be 1..3')
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ['BER_WORK_DIR'] = str(base)
    os.environ['BER_OUT_DIR'] = str(out/'output')
    sys.path.insert(0,str(code))
    import stage2 as s2
    import decision
    from data import load_ground_truth
    from prep import scan_norm
    from decide import write_lists
    meta = json.loads((base/'lgb_s2.json').read_text())
    if meta['tags'] != ['xenc'] or s2.P_MIN != .01:
        raise ValueError('Expected verified XENC baseline with P_MIN=0.01')
    cols = meta['cols']
    digest = hashlib.sha256(Path(__file__).read_bytes()+Path(mf.__file__).read_bytes()).hexdigest()
    baseline_hash = hashlib.sha256((base/'lgb_s2.json').read_bytes()).hexdigest()

    def context(scores, split, name):
        d = s2.build_ctx(scores,split,name,meta['tags'])
        if d['xenc'].null_count() or not d['xenc'].is_finite().all():
            raise ValueError(f'{name}: incomplete cached XENC scores')
        return d

    def join_pair_features(d, name):
        f = pl.read_parquet(base/'feat'/f'{name}.parquet',columns=KEYS+meta['stage1_cols'])
        joined = d.join(f,on=KEYS,how='inner',validate='1:1')
        if joined.height != d.height:
            raise ValueError(f'{name}: pair feature coverage mismatch')
        return joined

    def select_exact(kept,q):
        # Isotonic plateaus need deterministic raw-score/ID tie-breaking.
        d = kept.select(KEYS+['p2']).with_columns(pl.Series('q',q)).filter(pl.col('q')>=decision.P_FLOOR)
        d = d.sort(['s1','q','p2','cand'],descending=[False,True,True,False]).with_columns(
            pl.int_range(pl.len()).over('s1').alias('_r')).filter(pl.col('_r')<decision.N_MAX)
        if not d.height:
            return d.select(KEYS)
        d = d.with_columns((pl.col('s1').rank('dense')-1).cast(pl.Int64).alias('_i'))
        probabilities = np.zeros((int(d['_i'].max())+1,decision.N_MAX))
        probabilities[d['_i'].to_numpy(),d['_r'].to_numpy()] = d['q'].to_numpy()
        k = decision.exact_best_k(probabilities)
        return d.filter(pl.col('_r')<pl.Series(k[d['_i'].to_numpy()])).select(KEYS)

    def apply(d, dec):
        kept = owners(d)
        if dec['rule'] == 'threshold':
            return kept.filter(pl.col('p2') >= dec['thr']).select(KEYS)
        scores = kept['p2'].to_numpy()
        q = dec['calib'].predict(scores)
        for blank,iso in dec.get('by_blank',{}).items():
            mask = kept['ma_cand_blank'].to_numpy()==blank
            if mask.any():
                q[mask] = iso.predict(scores[mask])
        return select_exact(kept,q)

    def score(models, d, features):
        return d.with_columns(pl.Series('p2',s2._predict(models,s2._X(d,features))))

    if args.mode == 'predict':
        report = json.loads((out/'comparison.json').read_text())
        if report['code_sha256'] != digest or report['baseline_meta_sha256'] != baseline_hash:
            raise ValueError('Experiment code or baseline metadata changed since training')
        if (out/'PREDICT_STARTED.json').exists():
            raise FileExistsError('Prediction already exists; refusing overwrite')
        (out/'PREDICT_STARTED.json').write_text(json.dumps({'started':time.time()}))
        weights = json.loads((out/'name_weights.json').read_text())
        models = [lgb.Booster(model_file=str(out/f'variant_{i}.txt')) for i in range(report['seeds'])]
        baseline_models = [lgb.Booster(model_file=str(base/f'lgb_s2_{i}.txt')) for i in range(meta['n_models'])]
        d = context(pl.read_parquet(base/'test_scored.parquet'),'test','test')
        attrs = mf.read_attrs(base,'test',d)
        d = mf.add_features(d,attrs,weights)
        del attrs
        context_cols = [c for c in cols if c not in meta['stage1_cols']]+mf.EXTRA
        d = d.select(KEYS+context_cols)
        parts = []
        shards = sorted((base/'feat/test_shards').glob('*.parquet'))
        if not shards:
            raise FileNotFoundError('No test feature shards')
        for i,path in enumerate(shards):
            sh = pl.read_parquet(path,columns=KEYS+meta['stage1_cols']).join(d,on=KEYS,how='inner',validate='1:1')
            if sh.height:
                v = score(models,sh,cols+mf.EXTRA).select(KEYS+['p2','ma_cand_blank'])
                b = score(baseline_models,sh,cols).select(KEYS+['p2']).rename({'p2':'_baseline'})
                parts.append(v.join(b,on=KEYS,how='left',validate='1:1'))
            print(f'test scoring shard {i+1}/{len(shards)}',flush=True)
        te = pl.concat(parts)
        if te.height != d.height or te.select(KEYS).n_unique() != te.height:
            raise ValueError('Test scored coverage mismatch or duplicate pairs')
        te.write_parquet(out/'test_scored_micro.parquet')
        ids = scan_norm('test',1,['entity_id'])['entity_id'].to_list()
        target = out/'output'
        write_lists(ids,te.select(KEYS),target/'candidate_pairs.tsv','candidate_entity_ids')
        baseline = te.select(KEYS+[pl.col('_baseline').alias('p2')])
        files,seen_hashes = {},{}
        for name,data in five_scores(te.drop('_baseline'),baseline).items():
            with (out/f'{name}_decision.pkl').open('rb') as f:
                dec = pickle.load(f)
            final = apply(data,dec)
            if final['cand'].n_unique()!=final.height:
                raise ValueError('Candidate assigned to multiple Source-1 entities')
            path = target/f'matching_results_{name}.tsv'
            write_lists(ids,final,path,'matched_entity_ids')
            file_hash = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):
                    file_hash.update(block)
            sha = file_hash.hexdigest()
            files[name] = {'filename':path.name,'sha256':sha,'identical_to':seen_hashes.get(sha),
                          'passes_comparison':report['candidates'][name]['passes_comparison']}
            seen_hashes.setdefault(sha,name)
            print('GENERATED_MATCHING_TSV='+str(path),flush=True)
        ranking = report['candidate_ranking']
        (target/'validation_ranking.json').write_text(json.dumps({'ranking':ranking,'recommended':report['recommended'],'files':files,
            'status':'Exploratory reused-validation comparison; no leaderboard guarantee.'},indent=2))
        (out/'PREDICT_COMPLETE.json').write_text(json.dumps({'completed':time.time(),'candidates':CANDIDATES}))
        return

    val_ids = s2.s1_subset('train')['val'].to_list()
    cached_ids = set(pl.read_parquet(base/'cand/val.parquet',columns=['s1'])['s1'])
    if cached_ids!=set(val_ids):
        raise ValueError('Validation sample mismatch; check BER_N_TRAIN_S1/BER_N_VAL_S1')
    info = scan_norm('train',1,['entity_id','country','state','name_core']).filter(pl.col('entity_id').is_in(pl.Series(val_ids).implode()))
    roles,groups = partition(info)
    ids_by_role = {role:[s for s in val_ids if roles[s]==role] for role in ['stop','cal','hold']}
    print('Validation entity split: '+str({k:len(v) for k,v in ids_by_role.items()}),flush=True)
    split_counts = []
    for country in sorted(info['country'].unique()):
        row = {'country':country,'group_by':'state' if all(g[1].startswith('state:') for g in groups.values() if g[0]==country) else 'normalized_name'}
        row.update({role:sum(groups[s][0]==country for s in ids) for role,ids in ids_by_role.items()})
        split_counts.append(row)
    print('Split by country: '+json.dumps(split_counts),flush=True)
    fold_of = calibration_folds(info,ids_by_role['cal'])
    if args.mode=='preflight':
        print('PREFLIGHT_PASS: balanced split and calibration folds; training has not started.',flush=True)
        return
    if (out/'RUN_STARTED.json').exists():
        raise FileExistsError('Training already started here. Choose a fresh output directory.')
    (out/'RUN_STARTED.json').write_text(json.dumps({'started':time.time(),'code_sha256':digest}))
    tr = join_pair_features(context(pl.read_parquet(base/'s2_train_p.parquet').drop('fold'),
                                   'train','train'),'train')
    va = join_pair_features(context(pl.read_parquet(base/'s2_val_p.parquet'),'train','val'),'val')
    # Sampling settings are checked against both cached candidate and scored IDs.
    if not set(va['s1']).issubset(cached_ids):
        raise ValueError('Validation sample mismatch; check BER_N_TRAIN_S1/BER_N_VAL_S1')
    if set(tr['s1']) & set(val_ids):
        raise ValueError('Training/validation Source-1 overlap')
    truth = load_ground_truth().rename({'source1_entity_id':'s1','match':'cand'})
    truth = truth.filter(pl.col('s1').is_in(pl.Series(val_ids).implode()))
    (out/'split.json').write_text(json.dumps({'entity_ids':ids_by_role,
        'groups':{role:sorted({groups[s] for s in ids}) for role,ids in ids_by_role.items()},
        'counts_by_country':split_counts},indent=2))
    train_attrs = mf.read_attrs(base,'train',tr)
    weights = mf.fit_weights(train_attrs,tr['s1'].unique())
    (out/'name_weights.json').write_text(json.dumps(weights))
    tr = mf.add_features(tr,train_attrs,weights)
    del train_attrs
    val_attrs = mf.read_attrs(base,'train',va)
    va = mf.add_features(va,val_attrs,weights)
    del val_attrs
    # Persist added features for inspecting the error slices without recomputation.
    va.select(KEYS+mf.EXTRA).write_parquet(out/'val_micro_features.parquet')
    stop = va.filter(pl.col('s1').is_in(pl.Series(ids_by_role['stop']).implode()))
    if stop['y'].n_unique()!=2 or stop['s1'].n_unique()<.5*len(ids_by_role['stop']):
        raise ValueError('Stopping set lacks class coverage or sufficient scored entities')
    vectors, blank_diagnostics = {}, {}
    for name,features in [('control',cols),('variant',cols+mf.EXTRA)]:
        print(f'FIT {name}: {tr.height:,} rows, {len(features)} features, {args.seeds} seeds',flush=True)
        dt = lgb.Dataset(s2._X(tr,features),tr['y'].to_numpy(),feature_name=features)
        dv = lgb.Dataset(s2._X(stop,features),stop['y'].to_numpy(),reference=dt)
        models = []
        for seed in range(args.seeds):
            params = dict(s2.S2_PARAMS,seed=42+seed,bagging_seed=42+seed,feature_fraction_seed=42+seed)
            m = lgb.train(params,dt,4000,valid_sets=[dv],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(100)])
            m.save_model(str(out/f'{name}_{seed}.txt'))
            models.append(m)
        scored = score(models,va,features)
        scored.select(KEYS+['y','p','p2']).write_parquet(out/f'val_scored_{name}.parquet')
        cal_ids = ids_by_role['cal']
        # Preserve the same competing-owner graph used by final application.
        # Only calibration labels enter fitting; other states supply scores only.
        cal = owners(scored).filter(pl.col('s1').is_in(pl.Series(cal_ids).implode()))
        # Business-name groups stay together in balanced calibration folds.
        folds = np.array([fold_of[s] for s in cal['s1']])
        q = np.zeros(cal.height)
        for fold in (0,1):
            fit, valid = folds != fold, folds == fold
            if not fit.any() or not valid.any() or len(set(cal['y'].to_numpy()[fit])) != 2:
                raise ValueError('Calibration fold missing data or class')
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(cal['p2'].to_numpy()[fit],cal['y'].to_numpy()[fit])
            q[valid] = iso.predict(cal['p2'].to_numpy()[valid])
        grid = np.round(np.arange(.3,.961,.02),2)
        threshold, ft = max(((float(t),float(vector(cal.filter(pl.col('p2')>=t).select(KEYS),truth,cal_ids).mean()))
                             for t in grid),key=lambda x:x[1])
        fe = float(vector(select_exact(cal,q),truth,cal_ids).mean())
        iso = IsotonicRegression(out_of_bounds='clip').fit(cal['p2'].to_numpy(),cal['y'].to_numpy())
        dec = {'rule':'exact' if fe>=ft else 'threshold','thr':threshold,'calib':iso}
        with (out/f'{name}_decision.pkl').open('wb') as f:
            pickle.dump(dec,f)
        pred = apply(scored,dec)
        pred.write_parquet(out/f'val_pred_{name}.parquet')
        vectors[name] = vector(pred,truth,ids_by_role['hold'])
        blank = scored.filter((pl.col('ma_cand_blank') == 1) &
            pl.col('s1').is_in(pl.Series(ids_by_role['hold']).implode()))
        selected_blank = blank.join(pred,on=KEYS,how='semi')
        blank_diagnostics[name] = {
            'scored_pairs':blank.height, 'true_pairs':int(blank['y'].sum()),
            'selected_true':int(selected_blank['y'].sum()),
            'selected_false':int(selected_blank.height-selected_blank['y'].sum()),
            'missed_scored_true':int(blank['y'].sum()-selected_blank['y'].sum())}
        (out/f'{name}_summary.json').write_text(json.dumps({'decision':dec['rule'],'threshold':threshold,
            'cal_threshold_f05':ft,'cal_crossfit_exact_f05':fe,'hold_f05':float(vectors[name].mean()),
            'iterations':[m.best_iteration for m in models]},indent=2))
        print(f'{name} hold F0.5={vectors[name].mean():.9f}; decision={dec["rule"]}',flush=True)
        del dt,dv,models,scored
    hold = ids_by_role['hold']
    old = vector(pl.read_parquet(base/'val_pred_s2.parquet'),truth,hold)
    vs_control = comparison(vectors['control'],vectors['variant'],hold,groups)
    vs_baseline = comparison(old,vectors['variant'],hold,groups)
    blank_entities = set(va.filter(pl.col('ma_cand_blank') == 1)['s1'])
    slices = {}
    for name,mask in [('has_blank_candidate',np.array([s in blank_entities for s in hold])),
                      ('no_blank_candidate',np.array([s not in blank_entities for s in hold]))]:
        slices[name] = {'entities':int(mask.sum()),'delta_vs_control':float((vectors['variant']-vectors['control'])[mask].mean()) if mask.any() else None}
    # Five hypotheses are fixed before evaluating their held-out results.
    variant = pl.read_parquet(out/'val_scored_variant.parquet').join(
        va.select(KEYS+['ma_cand_blank']),on=KEYS,how='left',validate='1:1')
    baseline = pl.read_parquet(base/'val_scored_s2.parquet')
    candidate_reports = {}
    for name,data in five_scores(variant,baseline).items():
        cal = owners(data).filter(pl.col('s1').is_in(pl.Series(ids_by_role['cal']).implode()))
        iso = IsotonicRegression(out_of_bounds='clip').fit(cal['p2'].to_numpy(),cal['y'].to_numpy())
        dec = {'rule':'threshold' if name==CANDIDATES[1] else 'exact','thr':0.,'calib':iso}
        if name==CANDIDATES[1]:
            dec['thr'] = max(((float(t),float(vector(cal.filter(pl.col('p2')>=t).select(KEYS),truth,ids_by_role['cal']).mean()))
                             for t in np.round(np.arange(.3,.961,.02),2)),key=lambda x:x[1])[0]
        if name==CANDIDATES[2]:
            dec['by_blank'] = {}
            for blank in (0,1):
                sub = cal.filter(pl.col('ma_cand_blank')==blank)
                if sub.height>=200 and sub['y'].n_unique()==2:
                    dec['by_blank'][blank] = IsotonicRegression(out_of_bounds='clip').fit(sub['p2'].to_numpy(),sub['y'].to_numpy())
        with (out/f'{name}_decision.pkl').open('wb') as f:
            pickle.dump(dec,f)
        pred = apply(data,dec)
        pred.write_parquet(out/f'val_pred_{name}.parquet')
        new = vector(pred,truth,hold)
        control_cmp = comparison(vectors['control'],new,hold,groups)
        baseline_cmp = comparison(old,new,hold,groups)
        passed = bool(args.seeds==3 and all(c['delta']>=.0002 and c['group_bootstrap_ci95'][0]>0
            and all(v['delta']>=0 for v in c['country'].values()) for c in [control_cmp,baseline_cmp]))
        candidate_reports[name] = {'filename':f'matching_results_{name}.tsv','hold_f05':float(new.mean()),
            'versus_control':control_cmp,'versus_saved_baseline':baseline_cmp,
            'passes_comparison':passed,'decision':dec['rule'],
            'threshold':dec['thr'] if dec['rule']=='threshold' else None}
        print(f'CANDIDATE {name}: evaluation F0.5={new.mean():.9f}; passes={passed}',flush=True)
    ranking = sorted(CANDIDATES,key=lambda name:candidate_reports[name]['hold_f05'],reverse=True)
    passed = [name for name in ranking if candidate_reports[name]['passes_comparison']]
    eligible = bool(passed)
    report = {'baseline_public_score':.98076,'target_public_score':.99266666,
              'code_sha256':digest,'baseline_meta_sha256':baseline_hash,'seeds':args.seeds,
              'entities':{k:len(v) for k,v in ids_by_role.items()},'versus_control':vs_control,
              'versus_saved_baseline':vs_baseline,'slices':slices,
              'blank_candidate_diagnostics':blank_diagnostics,'any_candidate_passes_comparison':eligible,
              'generate_all_requested_candidates':True,
              'candidates':candidate_reports,'candidate_ranking':ranking,
              'recommended':passed[0] if passed else 'retain_verified_baseline',
              'split_counts_by_country':split_counts,
              'validation_status':'Exploratory comparison on reused validation with frozen upstream scores.',
              'limitations':['Stage-1 early stopping previously used validation; the saved baseline also used all validation.',
                 'New Stage-2 stopping and calibration exclude evaluation groups; India may use names when state balancing is impossible.',
                 'Five prespecified candidates are evaluated; their ranking is exploratory and reuses the evaluation set.',
                 'Few state groups make confidence intervals approximate. No French labels. No leaderboard guarantee.']}
    (out/'comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    print('COMPARISON_PASSED' if eligible else 'NO_ACCEPTED_IMPROVEMENT',flush=True)


if __name__ == '__main__':
    main()
