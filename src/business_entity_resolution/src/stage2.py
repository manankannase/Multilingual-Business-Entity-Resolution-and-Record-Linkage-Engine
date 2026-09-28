"""Stage 2: stage-1 probability + context / group features (+ GPU model scores) -> seed-bagged LightGBM -> decision.

  python stage2.py oof       # OOF stage-1 on train (2 state folds) + val scores -> s2_{train,val}_p.parquet
  (optional, GPU) gpu_xenc.py train 0/1, score train/val/test   -> xenc/<tag>_{train,val,test}.parquet
                  tags: xenc (encoder cross-encoder), llm (decoder LLM judge on the uncertain band)
  python stage2.py fit       # features, stage-2 models, decision layer tuned on val
  python stage2.py predict   # same on test (stage-1 scores from pipeline.predict), final TSV
BER_XENC_TAGS (default "xenc,llm") lists the GPU model scores to use when their files exist; set it to "" to
ignore them. BER_S2_SEEDS (default 3) = number of stage-2 LightGBM models averaged.
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import decision
from config import N_JOBS, OUT_DIR, SEED, WORK_DIR
from context import (CTX_COLS, CTX_STR, GROUP_COLS, context_features, expected_f05_select, group_features,
                     score_context, score_context_cols)
from data import load_ground_truth
from decide import one_to_one, sweep, write_lists
from pipeline import FEAT, s1_subset
from prep import scan_norm

P_MIN = float(os.environ.get("BER_P_MIN", 0.01))   # context over pairs with stage-1 p >= P_MIN (every split)
S1_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                 max_bin=127, num_threads=N_JOBS, verbose=-1, seed=SEED)
S2_PARAMS = dict(S1_PARAMS, num_leaves=63, learning_rate=0.03)
S2_SEEDS = int(os.environ.get("BER_S2_SEEDS", 3))
XDIR = WORK_DIR / "xenc"
TAG_PREFIX = {"xenc": "x", "llm": "l"}


def _X(df, cols):
    return df.select(cols).cast(pl.Float32).to_numpy()


def active_tags() -> list:
    tags = [t for t in os.environ.get("BER_XENC_TAGS", "xenc,llm").split(",") if t]
    return [t for t in tags if all((XDIR / f"{t}_{s}.parquet").exists() for s in ("train", "val", "test"))]


def tag_cols(tags) -> list:
    return [c for t in tags for c in [t] + score_context_cols(TAG_PREFIX.get(t, t))]


def cand_attrs(split: str, ids: pl.Series) -> pl.DataFrame:
    """Normalised fields of the given S2/S3 records (lazy filter: never loads the whole pool)."""
    frames = [pl.scan_parquet(WORK_DIR / "norm" / f"{split}_source{s}.parquet")
                .select(["entity_id"] + CTX_STR).filter(pl.col("entity_id").is_in(ids.implode())).collect()
              for s in (2, 3)]
    return pl.concat(frames)


def state_folds(tr: pl.DataFrame) -> np.ndarray:
    """2 folds of whole states (keeps competition between S1s intact)."""
    st = scan_norm("train", 1, ["entity_id", "state"]).rename({"entity_id": "s1"})
    tr_st = tr.select("s1").join(st, on="s1", how="left", maintain_order="left")["state"]
    states = sorted(tr_st.unique().to_list())
    rng = np.random.default_rng(SEED)
    fold_of = {s: int(rng.integers(2)) for s in states}
    return tr_st.replace_strict(fold_of, default=0).to_numpy()


def oof():
    """Stage-1 OOF scores on train (folds saved: the GPU models reuse them) and model scores on val."""
    meta = json.load(open(WORK_DIR / "lgb_v1.json"))
    cols, iters = meta["cols"], meta["iters"]
    tr = pl.read_parquet(FEAT / "train.parquet")
    fold = state_folds(tr)
    p = np.zeros(len(tr), np.float32)
    for k in (0, 1):
        trn, hold = fold != k, fold == k
        d = lgb.Dataset(_X(tr.filter(pl.Series(trn)), cols), tr["y"].to_numpy()[trn])
        m = lgb.train(S1_PARAMS, d, iters)
        p[hold] = m.predict(_X(tr.filter(pl.Series(hold)), cols), num_threads=N_JOBS)
        print(f"  OOF fold {k}: {hold.sum():,} pairs scored", flush=True)
    tr.select("s1", "cand", "y").with_columns(pl.Series("p", p), pl.Series("fold", fold.astype(np.int8))) \
      .write_parquet(WORK_DIR / "s2_train_p.parquet")
    del tr
    va = pl.read_parquet(FEAT / "val.parquet", columns=["s1", "cand", "y"] + cols)
    m1 = lgb.Booster(model_file=str(WORK_DIR / "lgb_v1.txt"))
    va.select("s1", "cand", "y").with_columns(pl.Series("p", m1.predict(_X(va, cols), num_threads=N_JOBS)
                                                          .astype(np.float32))) \
      .write_parquet(WORK_DIR / "s2_val_p.parquet")
    print("wrote s2_train_p.parquet / s2_val_p.parquet", flush=True)


def build_ctx(scored: pl.DataFrame, split: str, name: str, tags) -> pl.DataFrame:
    """scored: s1, cand, p (+ y). Returns pairs with p >= P_MIN and every context column."""
    t0 = time.time()
    keep = scored.filter(pl.col("p") >= P_MIN)
    attrs = cand_attrs(split, keep["cand"].unique())
    df = group_features(context_features(keep, attrs), attrs)
    for t in tags:
        x = pl.read_parquet(XDIR / f"{t}_{name}.parquet")
        df = df.join(x, on=["s1", "cand"], how="left")
        print(f"  {name}: {t} score present for {1 - df[t].is_null().mean():.1%} of pairs", flush=True)
        df = score_context(df, t, TAG_PREFIX.get(t, t))
    print(f"  {name}: context for {len(df):,} pairs in {time.time() - t0:.0f}s", flush=True)
    return df


def _predict(models, X) -> np.ndarray:
    return np.mean([m.predict(X, num_threads=N_JOBS) for m in models], axis=0).astype(np.float32)


def fit():
    meta = json.load(open(WORK_DIR / "lgb_v1.json"))
    cols = meta["cols"]
    tags = active_tags()
    tr = build_ctx(pl.read_parquet(WORK_DIR / "s2_train_p.parquet").drop("fold"), "train", "train", tags)
    va = build_ctx(pl.read_parquet(WORK_DIR / "s2_val_p.parquet"), "train", "val", tags)
    tr = tr.join(pl.read_parquet(FEAT / "train.parquet", columns=["s1", "cand"] + cols), on=["s1", "cand"])
    va = va.join(pl.read_parquet(FEAT / "val.parquet", columns=["s1", "cand"] + cols), on=["s1", "cand"])
    cols2 = cols + ["p"] + CTX_COLS + GROUP_COLS + tag_cols(tags)
    print(f"stage-2 data: train {len(tr):,} / val {len(va):,} pairs, {len(cols2)} features, "
          f"GPU scores {tags or 'none'}", flush=True)
    Xtr, Xva = _X(tr, cols2), _X(va, cols2)
    dtr = lgb.Dataset(Xtr, tr["y"].to_numpy(), feature_name=cols2, free_raw_data=False)
    dva = lgb.Dataset(Xva, va["y"].to_numpy(), reference=dtr)
    models = []
    for i in range(S2_SEEDS):          # seed bagging: different row / feature subsamples, averaged
        prm = dict(S2_PARAMS, seed=SEED + i, bagging_seed=SEED + i, feature_fraction_seed=SEED + i)
        m = lgb.train(prm, dtr, 4000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
        m.save_model(str(WORK_DIR / f"lgb_s2_{i}.txt"))
        models.append(m)
        print(f"  stage-2 model {i}: {m.best_iteration} iters", flush=True)
    va = va.with_columns(pl.Series("p2", _predict(models, Xva)))
    va.select("s1", "cand", "p", "p2", "y").write_parquet(WORK_DIR / "val_scored_s2.parquet")
    del Xtr, dtr

    val_ids = s1_subset("train")["val"]
    truth = load_ground_truth().rename({"source1_entity_id": "s1", "match": "cand"}) \
                               .filter(pl.col("s1").is_in(val_ids.implode()))
    grid = np.round(np.arange(0.2, 0.97, 0.01), 2)
    (t1, f1), _ = sweep(va.select("s1", "cand", "p"), truth, val_ids, grid, score="p")
    (t2, f2), _ = sweep(va.select("s1", "cand", "p2"), truth, val_ids, grid, score="p2")
    print(f"stage-1 only : F0.5 {f1:.4f} @ {t1}")
    print(f"stage-2      : F0.5 {f2:.4f} @ {t2}")
    imp = np.mean([m.feature_importance("gain") for m in models], axis=0)
    print("top stage-2 features:", [(c, int(g)) for c, g in sorted(zip(cols2, imp), key=lambda x: -x[1])[:25]])
    # decision layer: threshold vs singleton gate vs exact expected-F0.5 (cross-fitted on val state folds)
    state_of = dict(scan_norm("train", 1, ["entity_id", "state"]).filter(
        pl.col("entity_id").is_in(val_ids.implode())).iter_rows())
    summary, pred = decision.tune(va, truth, val_ids, state_of, score="p2")
    pred.write_parquet(WORK_DIR / "val_pred_s2.parquet")
    json.dump({"threshold": float(t2), "val_f05": summary[summary["chosen"]], "rule": "decision",
               "decision": summary, "cols": cols2, "stage1_cols": cols, "tags": tags, "n_models": S2_SEEDS,
               "stage1_val_f05": f1}, open(WORK_DIR / "lgb_s2.json", "w"))


def predict():
    meta = json.load(open(WORK_DIR / "lgb_s2.json"))
    models = [lgb.Booster(model_file=str(WORK_DIR / f"lgb_s2_{i}.txt")) for i in range(meta["n_models"])]
    tags = meta["tags"]
    missing = [t for t in tags if not (XDIR / f"{t}_test.parquet").exists()]
    if missing:
        sys.exit(f"stage-2 models use GPU scores {missing} but their test files are missing")
    scored = pl.read_parquet(WORK_DIR / "test_scored.parquet")          # s1, cand, p (stage 1, p >= 0.01)
    ctx = build_ctx(scored, "test", "test", tags)
    ctx_cols = [c for c in meta["cols"] if c not in meta["stage1_cols"]]
    ctx = ctx.select(["s1", "cand"] + ctx_cols)
    out = []
    for f in sorted((FEAT / "test_shards").glob("*.parquet")):
        sh = pl.read_parquet(f, columns=["s1", "cand"] + meta["stage1_cols"])
        sh = sh.join(ctx, on=["s1", "cand"], how="inner")
        if len(sh):
            keep = [c for c in decision.needed_cols() if c in sh.columns]      # S1-level gate inputs
            out.append(sh.select("s1", "cand", *keep).with_columns(
                pl.Series("p2", _predict(models, _X(sh, meta["cols"])))))
    te = pl.concat(out)
    te.write_parquet(WORK_DIR / "test_scored_s2.parquet")
    final = decision.apply(te)
    print(f"decision rule: {meta['decision']['chosen']}", flush=True)
    s1 = scan_norm("test", 1, ["entity_id", "country"])
    res = write_lists(s1["entity_id"].to_list(), final, OUT_DIR / "matching_results.tsv", "matched_entity_ids")
    n = res["matched_entity_ids"].str.len_chars()
    print(f"wrote {len(res):,} rows; empty {(n == 0).mean():.3f}; mean matches {final.height / len(res):.2f}")
    # per-country sanity check (France has no labels: its statistics should look like US / India)
    cnt = final.group_by("s1").len().rename({"s1": "entity_id", "len": "k"})
    print(s1.join(cnt, on="entity_id", how="left").with_columns(pl.col("k").fill_null(0))
            .group_by("country").agg(pl.len().alias("n_s1"), pl.col("k").mean().alias("mean_matches"),
                                     (pl.col("k") == 0).mean().alias("empty_rate")).sort("country"))


if __name__ == "__main__":
    {"oof": oof, "fit": fit, "predict": predict}[sys.argv[1]]()
