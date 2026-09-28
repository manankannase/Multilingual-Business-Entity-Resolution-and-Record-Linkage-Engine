"""Load challenge TSVs (tab-separated, no quoting) and cache them as parquet."""
import polars as pl

from config import DATA_DIR, WORK_DIR

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def load_source(split: str, src: int) -> pl.DataFrame:
    """split in {'train','test'}, src in {1,2,3}. Cached as parquet in WORK_DIR/raw."""
    cache = WORK_DIR / "raw" / f"{split}_source{src}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    df = read_tsv(DATA_DIR / split / f"{split}_source{src}.tsv").select(COLS).with_columns(
        pl.col(c).fill_null("") for c in COLS)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache)
    return df


def load_ground_truth() -> pl.DataFrame:
    """Long format: one row per (source1_entity_id, matched id); singletons have match = null."""
    cache = WORK_DIR / "raw" / "train_ground_truth_long.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    gt = read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv").with_columns(
        pl.col("matched_entity_ids").fill_null(""))
    long = (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("match"))
              .explode("match", empty_as_null=True)
              .with_columns(pl.when(pl.col("match") == "").then(None).otherwise(pl.col("match")).alias("match"))
              .select("source1_entity_id", "match"))
    cache.parent.mkdir(parents=True, exist_ok=True)
    long.write_parquet(cache)
    return long


def gt_sets(long: pl.DataFrame) -> dict:
    out = {}
    for s1, m in long.iter_rows():
        st = out.setdefault(s1, set())
        if m is not None:
            st.add(m)
    return out


if __name__ == "__main__":
    for split in ("train", "test"):
        for s in (1, 2, 3):
            print(split, s, load_source(split, s).shape)
    print("gt", load_ground_truth().shape)
