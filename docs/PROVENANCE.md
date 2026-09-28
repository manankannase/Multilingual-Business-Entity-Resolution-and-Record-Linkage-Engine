# Provenance and lineage

## Verified source archive

The exact predecessor archive is retained under `provenance/`:

| Packaged file | Original role | SHA-256 |
|---|---|---|
| `Edited_version_predecessor.zip` | User-identified predecessor on which the final edited work was built | `eac90123d4c4b259295f7b36be863decea85edf9761116d5895742479daa9c2c` |

File-by-file comparison found that the production tree under `src/` matches the business-entity-resolution source in `Edited_version_predecessor.zip`; only a generated `__pycache__` directory existed outside the archive.

## Separate TEJAS comparison

`TEJAS_v3.zip` was supplied later for comparison. It is not the predecessor that produced the verified final result. It differs from `Edited_version.zip` in at least:

- `src/stage2.py`
- `src/error_analysis.py`

The comparison experiment remains under `experiments/upgrade/` because it records a tested research path. The TEJAS archive itself is excluded to keep this repository's lineage unambiguous.

## Attribution boundary

- “User-provided” records how the files entered this work; it does not prove original authorship.
- Existing research notes and code inside predecessor archives retain their original filenames.
- This README does not assign individual contributions where commit-level evidence was unavailable.
- Add team names, repository history, and a chosen license before public release.

## Excluded material

- Competition datasets.
- Downloaded model weights.
- Virtual environments and caches.
- Large intermediate Parquet shards.
- Server paths, logs, and unrelated processes.

The exclusions keep the package distributable and avoid redistributing controlled data. Commands and source remain sufficient to explain and reproduce the pipeline when the legitimate dataset and model downloads are available.
