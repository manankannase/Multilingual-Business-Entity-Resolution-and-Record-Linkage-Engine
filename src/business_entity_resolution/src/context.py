"""Stage 2: context features computed over the whole scored candidate graph.

Input: pairs (s1, cand, p) where p is the stage-1 probability (out-of-fold on training data). Competition is
only meaningful when every S1 that could claim a record is present (test: all S1; train/val: whole states).
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

CTX_STR = ["name_core", "addr_clean", "house"]


def context_features(pairs: pl.DataFrame, cand_attrs: pl.DataFrame) -> pl.DataFrame:
    """pairs: s1, cand, p (+ anything). cand_attrs: entity_id + CTX_STR for candidate records."""
    p = pl.col("p")
    df = pairs.with_columns(
        # --- within S1 (how does this candidate compare with the S1's other candidates)
        p.rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias("c_rank_s1"),
        p.max().over("s1").alias("c_pmax_s1"),
        (p.max().over("s1") - p).alias("c_gap_s1"),
        (p >= 0.5).sum().over("s1").cast(pl.Int16).alias("c_n05_s1"),
        (p >= 0.2).sum().over("s1").cast(pl.Int16).alias("c_n02_s1"),
        p.sum().over("s1").alias("c_psum_s1"),
        pl.len().over("s1").cast(pl.Int16).alias("c_ncand_s1"),
        # --- across S1 (competition for the same S2/S3 record)
        pl.len().over("cand").cast(pl.Int16).alias("c_nclaim"),
        p.max().over("cand").alias("c_pmax_cand"),
        p.rank("ordinal", descending=True).over("cand").cast(pl.Int16).alias("c_rank_cand"),
        (p.sum().over("cand") - p).alias("c_pother_cand"),
    )
    # margin to the best OTHER S1 claiming this record
    second = (df.select("cand", "p").sort("p", descending=True)
                .group_by("cand", maintain_order=True).agg(pl.col("p").slice(1, 1).first().alias("_p2")))
    df = df.join(second, on="cand", how="left").with_columns(
        pl.when(pl.col("c_rank_cand") == 1).then(p - pl.col("_p2").fill_null(0.0))
          .otherwise(p - pl.col("c_pmax_cand")).alias("c_margin_cand")).drop("_p2")
    # --- agreement with the S1's top candidate (duplicates of one business resemble each other)
    top = (df.filter(pl.col("c_rank_s1") == 1).select("s1", pl.col("cand").alias("top_cand"),
                                                       pl.col("p").alias("_ptop")))
    df = df.join(top, on="s1", how="left")
    a = df.select(pl.col("cand").alias("entity_id")).join(cand_attrs, on="entity_id", how="left",
                                                          maintain_order="left")
    b = df.select(pl.col("top_cand").alias("entity_id")).join(cand_attrs, on="entity_id", how="left",
                                                              maintain_order="left")
    an, bn = a["name_core"].fill_null("").to_list(), b["name_core"].fill_null("").to_list()
    aa, ba = a["addr_clean"].fill_null("").to_list(), b["addr_clean"].fill_null("").to_list()
    is_top = (df["cand"] == df["top_cand"]).to_numpy()
    n_sim = cpdist(an, bn, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    a_sim = cpdist(aa, ba, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    h1, h2 = a["house"].fill_null("").to_numpy(), b["house"].fill_null("").to_numpy()
    h_eq = np.where((h1 == "") | (h2 == ""), -1, (h1 == h2).astype(np.int8)).astype(np.int8)
    n_sim[is_top] = np.nan
    a_sim[is_top] = np.nan
    h_eq[is_top] = -2
    return df.with_columns(pl.Series("c_top_name_sim", n_sim), pl.Series("c_top_addr_sim", a_sim),
                           pl.Series("c_top_house_eq", h_eq)).drop("top_cand", "_ptop")


CTX_COLS = ["c_rank_s1", "c_pmax_s1", "c_gap_s1", "c_n05_s1", "c_n02_s1", "c_psum_s1", "c_ncand_s1",
            "c_nclaim", "c_pmax_cand", "c_rank_cand", "c_pother_cand", "c_margin_cand",
            "c_top_name_sim", "c_top_addr_sim", "c_top_house_eq"]


GROUP_K = 4
GROUP_COLS = ["t_max_name_sim", "t_max_addr_sim", "t_wmean_name_sim", "t_wmean_addr_sim", "t_support",
              "t_house_agree", "t_n_others"]


def group_features(df: pl.DataFrame, cand_attrs: pl.DataFrame, chunk: int = 4_000_000) -> pl.DataFrame:
    """Collective evidence: agreement of each candidate with the S1's OTHER top-GROUP_K candidates (by stage-1 p).
    The S2/S3 duplicates of one business resemble each other; a same-name decoy at another address does not.
    t_support = sum of p over other top candidates that agree on both name and address (>= 80 token-set)."""
    top = df.filter(pl.col("c_rank_s1") <= GROUP_K).select(
        "s1", pl.col("cand").alias("o"), pl.col("p").alias("po"))
    x = (df.select("s1", "cand").join(top, on="s1").filter(pl.col("cand") != pl.col("o")))
    attrs = cand_attrs.select("entity_id", "name_core", "addr_clean", "house")
    parts = []
    for s in range(0, len(x), chunk):
        c = x.slice(s, chunk)
        a = c.select(pl.col("cand").alias("entity_id")).join(attrs, on="entity_id", how="left",
                                                              maintain_order="left")
        b = c.select(pl.col("o").alias("entity_id")).join(attrs, on="entity_id", how="left", maintain_order="left")
        n_sim = cpdist(a["name_core"].fill_null("").to_list(), b["name_core"].fill_null("").to_list(),
                       scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        a_sim = cpdist(a["addr_clean"].fill_null("").to_list(), b["addr_clean"].fill_null("").to_list(),
                       scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        h1, h2 = a["house"].fill_null("").to_numpy(), b["house"].fill_null("").to_numpy()
        h_eq = np.where((h1 == "") | (h2 == ""), 0.5, (h1 == h2).astype(np.float32)).astype(np.float32)
        parts.append(c.select("s1", "cand", "po").with_columns(
            pl.Series("ns", n_sim), pl.Series("as", a_sim), pl.Series("he", h_eq)))
        del a, b, n_sim, a_sim
    if not parts:
        return df.with_columns([pl.lit(None, pl.Float32).alias(c) for c in GROUP_COLS])
    g = (pl.concat(parts).group_by("s1", "cand").agg(
        pl.col("ns").max().alias("t_max_name_sim"),
        pl.col("as").max().alias("t_max_addr_sim"),
        ((pl.col("ns") * pl.col("po")).sum() / pl.col("po").sum().clip(1e-6)).alias("t_wmean_name_sim"),
        ((pl.col("as") * pl.col("po")).sum() / pl.col("po").sum().clip(1e-6)).alias("t_wmean_addr_sim"),
        (pl.col("po") * ((pl.col("ns") >= 80) & (pl.col("as") >= 80)).cast(pl.Float32)).sum().alias("t_support"),
        pl.col("he").max().alias("t_house_agree"),
        pl.len().cast(pl.Int16).alias("t_n_others")))
    return df.join(g, on=["s1", "cand"], how="left")


def score_context(df: pl.DataFrame, col: str, prefix: str) -> pl.DataFrame:
    """Rank / gap of any pair score inside its S1 and among the S1s competing for the same candidate."""
    s = pl.col(col)
    df = df.with_columns(
        s.rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias(f"{prefix}_rank_s1"),
        (s.max().over("s1") - s).alias(f"{prefix}_gap_s1"),
        s.rank("ordinal", descending=True).over("cand").cast(pl.Int16).alias(f"{prefix}_rank_cand"),
        s.max().over("cand").alias("_smax"))
    second = (df.select("cand", col).sort(col, descending=True, nulls_last=True)
                .group_by("cand", maintain_order=True).agg(pl.col(col).slice(1, 1).first().alias("_s2")))
    return (df.join(second, on="cand", how="left")
              .with_columns(pl.when(pl.col(f"{prefix}_rank_cand") == 1).then(s - pl.col("_s2"))
                              .otherwise(s - pl.col("_smax")).alias(f"{prefix}_margin_cand"))
              .drop("_smax", "_s2"))


def score_context_cols(prefix: str):
    return [f"{prefix}_rank_s1", f"{prefix}_gap_s1", f"{prefix}_rank_cand", f"{prefix}_margin_cand"]


# ---------------------------------------------------------------------------------- decision rule
def expected_f05_select(pairs: pl.DataFrame, score: str = "p", min_p: float = 0.05, beta2: float = 0.25):
    """Per S1: sort candidates by calibrated p and choose k maximising E[F0.5] (ratio-of-expectations
    approximation: E[F] ≈ (1+b²)·Σ_top-k p / (b²·E|true| + k)); k = 0 scores P(no match) ≈ Π(1-p)."""
    d = (pairs.filter(pl.col(score) >= min_p).sort(["s1", score], descending=[False, True])
              .with_columns(pl.col(score).cum_sum().over("s1").alias("_cs"),
                            pl.int_range(1, pl.len() + 1).over("s1").alias("_k"),
                            pl.col(score).sum().over("s1").alias("_et"),
                            (1 - pl.col(score)).log().sum().over("s1").exp().alias("_p0")))
    d = d.with_columns(((1 + beta2) * pl.col("_cs") / (beta2 * pl.col("_et") + pl.col("_k"))).alias("_ef"))
    best = d.group_by("s1").agg(pl.col("_ef").max().alias("_best"), pl.col("_p0").first())
    d = d.join(best, on="s1").with_columns(
        (pl.col("_ef") == pl.col("_best")).cast(pl.Int8).alias("_isbest"))
    kbest = d.filter(pl.col("_isbest") == 1).group_by("s1").agg(pl.col("_k").min().alias("_kbest"),
                                                                 pl.col("_best").first(), pl.col("_p0").first())
    d = d.join(kbest.select("s1", "_kbest", pl.col("_best").alias("_bestv")), on="s1")
    chosen = d.filter((pl.col("_k") <= pl.col("_kbest")) & (pl.col("_bestv") > pl.col("_p0")))
    return chosen.select("s1", "cand")
