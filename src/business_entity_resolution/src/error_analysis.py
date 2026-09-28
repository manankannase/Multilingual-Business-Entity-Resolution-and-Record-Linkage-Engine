"""Where does the validation F0.5 go? Decomposes the loss into blocking misses / false positives / false negatives.

  python error_analysis.py stage1     # after pipeline.py fit
  python error_analysis.py stage2     # after stage2.py fit

"Gain if fixed" = macro F0.5 with that one error type removed (all else unchanged): it tells you which part of
the pipeline is worth working on next.
"""
import json
import sys

import numpy as np
import polars as pl

from config import WORK_DIR
from context import expected_f05_select
from data import load_ground_truth
from decide import one_to_one
from metric import entity_f05
from pipeline import CAND, s1_subset
from prep import scan_norm


def _sets(df: pl.DataFrame) -> dict:
    out = {}
    for s, c in df.select("s1", "cand").iter_rows():
        if c is not None:
            out.setdefault(s, set()).add(c)
    return out


def main(stage: str):
    if stage == "stage1":
        meta = json.load(open(WORK_DIR / "lgb_v1.json"))
        va = pl.read_parquet(WORK_DIR / "val_scored.parquet")
        score, rule = "p", "threshold"
    else:
        meta = json.load(open(WORK_DIR / "lgb_s2.json"))
        va = pl.read_parquet(WORK_DIR / "val_scored_s2.parquet")
        score, rule = "p2", meta["rule"]
    if rule == "decision":                  # cross-fitted predictions of the chosen decision rule
        pred = pl.read_parquet(WORK_DIR / "val_pred_s2.parquet")
        rule = f"decision:{meta['decision']['chosen']}"
    else:
        o = one_to_one(va.select("s1", "cand", score), score)
        pred = expected_f05_select(o, score=score) if rule == "expected" else \
            o.filter(pl.col(score) >= meta["threshold"]).select("s1", "cand")

    ids = s1_subset("train")["val"].to_list()
    truth = load_ground_truth().rename({"source1_entity_id": "s1", "match": "cand"}) \
                               .filter(pl.col("s1").is_in(pl.Series(ids).implode()))
    cand = pl.read_parquet(CAND / "val.parquet", columns=["s1", "cand"])
    P, T, C = _sets(pred), _sets(truth), _sets(cand)
    ctry = dict(scan_norm("train", 1, ["entity_id", "country"]).filter(
        pl.col("entity_id").is_in(pl.Series(ids).implode())).iter_rows())

    def macro(fn):
        return float(np.mean([entity_f05(fn(i), T.get(i, set())) for i in ids]))

    e = set()
    cur = macro(lambda i: P.get(i, e))
    no_fp = macro(lambda i: P.get(i, e) & T.get(i, e))
    no_fn = macro(lambda i: P.get(i, e) | (T.get(i, e) & C.get(i, e)))
    no_block = macro(lambda i: P.get(i, e) | (T.get(i, e) - C.get(i, e)))
    perfect_given_blocking = macro(lambda i: T.get(i, e) & C.get(i, e))

    n_true = sum(len(v) for v in T.values())
    n_miss = sum(len(T.get(i, e) - C.get(i, e)) for i in ids)
    n_fp = sum(len(P.get(i, e) - T.get(i, e)) for i in ids)
    n_fn = sum(len((T.get(i, e) & C.get(i, e)) - P.get(i, e)) for i in ids)
    print(f"=== {stage} error analysis on {len(ids):,} val S1 ({rule}, score {score})")
    print(f"macro F0.5 now                         : {cur:.4f}")
    print(f"ceiling with perfect matcher (blocking): {perfect_given_blocking:.4f}")
    print(f"gain if no false positives             : +{no_fp - cur:.4f}   ({n_fp:,} FP pairs)")
    print(f"gain if no FN among candidates         : +{no_fn - cur:.4f}   ({n_fn:,} FN pairs)")
    print(f"gain if blocking missed nothing        : +{no_block - cur:.4f}   ({n_miss:,}/{n_true:,} true pairs "
          f"never candidates, recall {1 - n_miss / max(n_true, 1):.4f})")

    # entity-level buckets
    rows = []
    for i in ids:
        t, p = T.get(i, e), P.get(i, e)
        f = entity_f05(p, t)
        kind = ("singleton_ok" if not t and not p else "singleton_FP" if not t else
                "matched_empty_pred" if not p else "exact" if p == t else
                "has_FP" if p - t else "only_FN")
        rows.append((ctry.get(i, "?"), kind, 1 - f))
    df = pl.DataFrame(rows, schema=["country", "kind", "loss"], orient="row")
    n = len(ids)
    print("\nloss by entity bucket (share of the total F0.5 lost):")
    print(df.group_by("kind").agg(pl.len().alias("n_s1"), (pl.col("loss").sum() / n).alias("F_lost"))
            .sort("F_lost", descending=True))
    print("\nper country:")
    print(df.group_by("country").agg(pl.len().alias("n_s1"), (1 - pl.col("loss").mean()).alias("F05"),
                                     (pl.col("loss").sum() / n).alias("F_lost_of_total")).sort("country"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "stage1")
