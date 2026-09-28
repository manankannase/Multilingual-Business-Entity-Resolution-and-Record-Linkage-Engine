"""Conservative data preparation; text groups are proxies, not true entity IDs."""
import hashlib
import polars as pl


def bucket(text):
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little") % 100


def prepare(path, limit=0):
    d = pl.read_parquet(path)
    if not {"a", "b", "label", "origin"} <= set(d.columns):
        raise ValueError("Expected a, b, label, origin columns")
    d = d.select("a", "b", "label", "origin")
    if any(d.null_count().row(0)) or not set(d["label"].unique()) <= {0, 1}:
        raise ValueError("Null inputs or non-binary labels")
    if not set(d["origin"].unique()) <= {"train", "fr_pseudo"}:
        raise ValueError("Unknown origin; never silently treat predictions as truth")
    # Conflicting exact text pairs cannot be disambiguated with this file's fields.
    bad = d.group_by("a", "b").agg(pl.col("label").n_unique().alias("n")).filter(pl.col("n") > 1)
    d = d.join(bad.select("a", "b"), on=["a", "b"], how="anti")
    # Prefer true labels over duplicate pseudo-labels.
    d = d.with_columns((pl.col("origin") == "train").alias("_true"))
    d = d.sort("_true", descending=True, maintain_order=True).unique(["a", "b"], maintain_order=True).drop("_true")
    groups = d.select("a").unique().with_columns(pl.col("a").map_elements(bucket, return_dtype=pl.Int64).alias("_group"))
    d = d.join(groups, on="a", how="left", maintain_order="left")
    genuine = d.filter(pl.col("origin") == "train")
    dev = genuine.filter(pl.col("_group") < 2)
    audit = genuine.filter((pl.col("_group") >= 2) & (pl.col("_group") < 4))
    held = pl.concat([dev, audit]).select("a").unique()
    fit = d.join(held, on="a", how="anti")
    # Also exclude exact target-text reuse from validation, including pseudo data.
    targets = pl.concat([dev, audit]).select("b").unique()
    fit = fit.join(targets, on="b", how="anti")
    if limit:
        fit = fit.sample(n=min(limit, fit.height), seed=17, shuffle=True)
        dev = dev.head(256)
        audit = audit.head(256)
    if not fit.height or any(set(x["label"].unique()) != {0, 1} for x in (dev, audit)):
        raise ValueError("Need nonempty training and both labels in dev/audit")
    return fit, dev, audit, {"conflicting_text_pairs_removed": bad.height,
        "fit": fit.height, "dev_true_only": dev.height, "audit_true_only": audit.height,
        "fit_origins": dict(fit["origin"].value_counts().iter_rows()),
        "limitation": "No entity IDs: text separation cannot guarantee entity-disjoint evaluation; pair metrics are not leaderboard macro F0.5."}
