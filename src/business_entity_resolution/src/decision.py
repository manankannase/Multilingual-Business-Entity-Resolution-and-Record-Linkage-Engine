"""Final decision layer: which of an S1's scored candidates go into matching_results.tsv.

Candidates (after one-to-one) are compared under several rules on validation, and the best one is kept:
  threshold      : p2 >= t                                  (previous behaviour)
  gate+threshold : as above, but S1s whose P(has a match) < g get an empty list
  exact          : per S1, choose the top-k maximising the EXACT expected F0.5 under independent calibrated
                   probabilities (Poisson-binomial over the chosen / unchosen candidates; Jansche 2007,
                   Dembczynski et al. NeurIPS 2011). k = 0 is worth P(no true match) = prod(1 - q)
  exact+gate     : same, but P(no true match) comes from the S1-level gate model (singleton detector)
Calibration (isotonic) and the gate (small LightGBM on S1-level features) are cross-fitted on 2 folds of whole
states for the comparison, then refit on all of validation for test.
"""
import pickle
import zlib

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from config import N_JOBS, SEED, WORK_DIR
from decide import evaluate, one_to_one

DEC_PATH = WORK_DIR / "decision.pkl"
BETA2 = 0.25
N_MAX = 20                # candidates per S1 considered by the exact rule (after one-to-one)
P_FLOOR = 0.01            # calibrated probability below which a candidate is never chosen
TOP_COLS = ["p", "c_margin_cand", "c_nclaim", "c_top_addr_sim", "c_ncand_s1", "n_s1_same_name",
            "t_support", "t_wmean_addr_sim", "xenc", "x_margin_cand", "x_gap_s1", "llm", "l_margin_cand"]
GATE_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                   feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   num_threads=N_JOBS, verbose=-1, seed=SEED)
GATE_ROUNDS = 400


def needed_cols():
    return TOP_COLS


# ------------------------------------------------------------------------------------------ S1-level table
def s1_table(pairs: pl.DataFrame, kept: pl.DataFrame, score: str = "p2") -> pl.DataFrame:
    """One row per S1 that has at least one scored pair: features of its candidate list (g_*) + label y_any."""
    tops = [c for c in TOP_COLS if c in pairs.columns]
    all_ = pairs.group_by("s1").agg(pl.col(score).max().alias("g_max_all"), pl.len().alias("g_n_all"))
    k = (kept.sort(score, descending=True)
             .group_by("s1", maintain_order=True)
             .agg(pl.col(score).first().alias("g_max"),
                  pl.col(score).slice(1, 1).first().alias("g_2nd"),
                  pl.col(score).slice(2, 1).first().alias("g_3rd"),
                  pl.col(score).sum().alias("g_sum"),
                  (pl.col(score) >= 0.5).sum().alias("g_n05"),
                  (pl.col(score) >= 0.2).sum().alias("g_n02"),
                  pl.len().alias("g_n_kept"),
                  *[pl.col(c).first().alias(f"g_top_{c}") for c in tops],
                  *([pl.col("y").max().alias("y_any")] if "y" in kept.columns else [])))
    t = all_.join(k, on="s1", how="left").with_columns(
        (pl.col("g_max_all") - pl.col("g_max").fill_null(0)).alias("g_lost_to_other_s1"),
        (pl.col("g_n_all") - pl.col("g_n_kept").fill_null(0)).alias("g_n_lost"))
    if "y_any" in t.columns:
        t = t.with_columns(pl.col("y_any").fill_null(0))
    return t


def gate_feats(t: pl.DataFrame):
    return [c for c in t.columns if c.startswith("g_")]


def _fit_gate(t: pl.DataFrame, feats):
    d = lgb.Dataset(t.select(feats).cast(pl.Float32).to_numpy(), t["y_any"].to_numpy())
    return lgb.train(GATE_PARAMS, d, GATE_ROUNDS)


def _gate_prob(model, t: pl.DataFrame, feats) -> np.ndarray:
    return model.predict(t.select(feats).cast(pl.Float32).to_numpy(), num_threads=N_JOBS)


def _fit_calib(kept: pl.DataFrame, score: str):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(kept[score].to_numpy(), kept["y"].to_numpy())
    return iso


# ------------------------------------------------------------------------------------------ rules
def exact_best_k(P: np.ndarray, p_empty: np.ndarray = None, chunk: int = 20000) -> np.ndarray:
    """P: (S, n) calibrated probabilities sorted descending per row (0-padded). Returns the k (0..n) that
    maximises E[F_beta] of predicting the top-k, assuming independent candidates:
      E = sum_{a,b} Pr(TP=a among top-k) Pr(b positives among the rest) (1+b2) a / (b2 (a+b) + k),
    k = 0 scores Pr(no positives) (or 1 - gate probability when p_empty is given)."""
    S, n = P.shape
    best = np.zeros(S, np.int16)
    a = np.arange(n + 1, dtype=np.float64)[:, None]
    b = np.arange(n + 1, dtype=np.float64)[None, :]
    for s0 in range(0, S, chunk):
        p = P[s0:s0 + chunk].astype(np.float64)
        m = len(p)
        Q = np.zeros((n + 1, m, n + 1))                 # Q[j] = distribution of #positives in items j..n-1
        Q[n][:, 0] = 1.0
        for j in range(n - 1, -1, -1):
            pj = p[:, j:j + 1]
            Q[j] = Q[j + 1] * (1 - pj)
            Q[j][:, 1:] += Q[j + 1][:, :-1] * pj
        vals = np.empty((m, n + 1))
        vals[:, 0] = Q[0][:, 0] if p_empty is None else p_empty[s0:s0 + m]
        A = np.zeros((m, n + 1))                        # distribution of TP among the first k items
        A[:, 0] = 1.0
        for k in range(1, n + 1):
            pk = p[:, k - 1:k]
            A2 = A * (1 - pk)
            A2[:, 1:] += A[:, :-1] * pk
            A = A2
            W = (1 + BETA2) * a / (BETA2 * (a + b) + k)
            vals[:, k] = np.einsum("ma,ab,mb->m", A, W, Q[k], optimize=True)
        best[s0:s0 + m] = vals.argmax(1)
    return best


def select_exact(kept: pl.DataFrame, q: np.ndarray, p_empty: dict = None) -> pl.DataFrame:
    """kept: one-to-one pairs; q: calibrated probability per row; p_empty: {s1: P(no match)} or None."""
    d = (kept.select("s1", "cand").with_columns(pl.Series("q", q.astype(np.float64)))
             .filter(pl.col("q") >= P_FLOOR)
             .sort(["s1", "q"], descending=[False, True])
             .with_columns(pl.int_range(pl.len()).over("s1").alias("_r"))
             .filter(pl.col("_r") < N_MAX)
             .with_columns((pl.col("s1").rank("dense") - 1).cast(pl.Int64).alias("_i")))
    if len(d) == 0:
        return d.select("s1", "cand")
    S = int(d["_i"].max()) + 1
    P = np.zeros((S, N_MAX))
    P[d["_i"].to_numpy(), d["_r"].to_numpy()] = d["q"].to_numpy()
    pe = None
    if p_empty is not None:
        s1_of = d.group_by("_i").agg(pl.col("s1").first()).sort("_i")["s1"].to_list()
        pe = np.array([p_empty.get(s, 1.0) for s in s1_of])
    k = exact_best_k(P, pe)
    return d.filter(pl.col("_r") < pl.Series(k[d["_i"].to_numpy()])).select("s1", "cand")


def select_threshold(kept: pl.DataFrame, thr: float, score: str) -> pl.DataFrame:
    return kept.filter(pl.col(score) >= thr).select("s1", "cand")


def _gated(pred: pl.DataFrame, gate: dict, g: float) -> pl.DataFrame:
    ok = [s for s, v in gate.items() if v >= g]
    return pred.filter(pl.col("s1").is_in(pl.Series(ok, dtype=pl.Utf8).implode()))


# ------------------------------------------------------------------------------------------ tune / apply
def tune(va: pl.DataFrame, truth: pl.DataFrame, val_ids, state_of: dict, score: str = "p2"):
    """va: every scored val pair (s1, cand, score, y, context columns). Returns (summary dict, val predictions)."""
    kept = one_to_one(va, score)
    tab = s1_table(va, kept, score)
    feats = gate_feats(tab)
    fold_s1 = {s: zlib.crc32(str(state_of.get(s, "")).encode()) % 2 for s in tab["s1"].to_list()}
    tab = tab.with_columns(pl.Series("_f", [fold_s1[s] for s in tab["s1"].to_list()]))
    kept = kept.with_columns(pl.col("s1").replace_strict(fold_s1, default=0).alias("_f"))

    # cross-fitted gate probabilities and calibrated probabilities
    gate_cf = np.zeros(len(tab))
    q_cf = np.zeros(len(kept))
    for f in (0, 1):
        tr_t, ho_t = tab["_f"].to_numpy() != f, tab["_f"].to_numpy() == f
        m = _fit_gate(tab.filter(pl.Series(tr_t)), feats)
        gate_cf[ho_t] = _gate_prob(m, tab.filter(pl.Series(ho_t)), feats)
        tr_k, ho_k = kept["_f"].to_numpy() != f, kept["_f"].to_numpy() == f
        iso = _fit_calib(kept.filter(pl.Series(tr_k)), score)
        q_cf[ho_k] = iso.predict(kept.filter(pl.Series(ho_k))[score].to_numpy())
    gate = dict(zip(tab["s1"].to_list(), gate_cf))
    y_any = tab["y_any"].to_numpy()
    print(f"  gate (S1 has a match among candidates): positives {y_any.mean():.3f}, "
          f"cross-fitted accuracy@0.5 {np.mean((gate_cf >= 0.5) == (y_any == 1)):.4f}", flush=True)

    F = lambda pred: evaluate(pred, truth, val_ids)
    res = {}
    grid = np.round(np.arange(0.2, 0.97, 0.02), 2)
    th = [(t, F(select_threshold(kept, t, score))) for t in grid]
    t_best, f_thr = max(th, key=lambda x: x[1])
    res["threshold"] = (f_thr, {"thr": float(t_best)})
    best_g = max(((g, F(_gated(select_threshold(kept, t_best, score), gate, g)))
                  for g in np.round(np.arange(0.1, 0.91, 0.05), 2)), key=lambda x: x[1])
    res["gate+threshold"] = (best_g[1], {"thr": float(t_best), "g": float(best_g[0])})
    res["exact"] = (F(select_exact(kept, q_cf)), {})
    res["exact+gate"] = (F(select_exact(kept, q_cf, {s: 1 - v for s, v in gate.items()})), {})
    for name, (f, prm) in res.items():
        print(f"  decision {name:15s}: val F0.5 {f:.4f} {prm}", flush=True)
    rule = max(res, key=lambda r: res[r][0])

    # refit on all of validation for test
    dec = {"rule": rule, "score": score, **res[rule][1], "feats": feats, "val_f05": res[rule][0],
           "gate": _fit_gate(tab, feats).model_to_string(), "calib": _fit_calib(kept, score)}
    with open(DEC_PATH, "wb") as fh:
        pickle.dump(dec, fh)
    print(f"  chosen decision rule: {rule} (val F0.5 {res[rule][0]:.4f}) -> {DEC_PATH}", flush=True)
    pred = {"threshold": lambda: select_threshold(kept, t_best, score),
            "gate+threshold": lambda: _gated(select_threshold(kept, t_best, score), gate, best_g[0]),
            "exact": lambda: select_exact(kept, q_cf),
            "exact+gate": lambda: select_exact(kept, q_cf, {s: 1 - v for s, v in gate.items()})}[rule]()
    return {r: v[0] for r, v in res.items()} | {"chosen": rule}, pred


def apply(te: pl.DataFrame) -> pl.DataFrame:
    """te: every scored test pair (s1, cand, score + context columns). Returns the chosen (s1, cand) pairs."""
    with open(DEC_PATH, "rb") as fh:
        dec = pickle.load(fh)
    score = dec["score"]
    kept = one_to_one(te, score)
    rule = dec["rule"]
    gate = None
    if "gate" in rule:
        tab = s1_table(te, kept, score)
        for c in dec["feats"]:                    # a column missing on test (should not happen) -> null
            if c not in tab.columns:
                tab = tab.with_columns(pl.lit(None, pl.Float32).alias(c))
        prob = _gate_prob(lgb.Booster(model_str=dec["gate"]), tab, dec["feats"])
        gate = dict(zip(tab["s1"].to_list(), prob))
    if rule == "threshold":
        return select_threshold(kept, dec["thr"], score)
    if rule == "gate+threshold":
        return _gated(select_threshold(kept, dec["thr"], score), gate, dec["g"])
    q = dec["calib"].predict(kept[score].to_numpy())
    return select_exact(kept, q, None if rule == "exact" else {s: 1 - v for s, v in gate.items()})
