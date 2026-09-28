# Technical references

These references informed the architecture or evaluation logic. The implementation uses supplied competition data only; citations are technical background, not external entity data.

## Entity matching

1. **Ditto: Deep Entity Matching with Pre-Trained Language Models** — Li et al., VLDB 2020. [Paper](https://arxiv.org/abs/2004.00584)  
   Motivation: jointly encode both records so a transformer can learn cross-field and cross-token interactions. This supports the pairwise cross-encoder design in `gpu_xenc.py`.

2. **DeepMatcher: A Deep Learning Approach to Entity Matching** — Mudgal et al., SIGMOD 2018. [Paper](https://arxiv.org/abs/1710.00597)  
   Motivation: field-aware representation and difficult negative examples. It informed separating name/address signals and prioritizing hard negatives.

3. **Entity Resolution: Theory, Practice & Open Challenges** — Christophides et al., PVLDB 2021. [Paper](https://arxiv.org/abs/2010.11049)  
   Motivation: candidate generation, matching, clustering, and evaluation are distinct error sources. The project audits retrieval, scoring, and final decision separately.

## Retrieval and multilingual representations

4. **E5 Text Embeddings** — Wang et al., 2022. [Paper](https://arxiv.org/abs/2212.03533)  
   Motivation: contrastively trained retrieval representations. `gpu_dense.py` uses multilingual E5-small for dense candidate recall; `gpu_xenc.py` initializes from multilingual E5-base.

5. **Multilingual-E5** — Wang et al., 2024. [Paper](https://arxiv.org/abs/2402.05672) · [Model card](https://huggingface.co/intfloat/multilingual-e5-base)  
   Motivation: shared multilingual semantic representation across English, Indic scripts, and French. It complements character n-gram retrieval when strings differ substantially.

6. **Efficient sparse top-n multiplication (`sparse-dot-topn`)** — [Project](https://github.com/ing-bank/sparse_dot_topn)  
   Motivation: compute only highest TF-IDF similarities instead of materializing the full pairwise matrix.

## Models and calibration

7. **LightGBM: A Highly Efficient Gradient Boosting Decision Tree** — Ke et al., NeurIPS 2017. [Paper](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)  
   Motivation: fast nonlinear learning over millions of mixed similarity and context features.

8. **Predicting Good Probabilities with Supervised Learning** — Niculescu-Mizil and Caruana, ICML 2005. [Paper](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)  
   Motivation: ranking quality and probability quality differ. Calibration is required before expected-utility decoding.

9. **Thresholding Classifiers to Maximize F1 Score** — Lipton et al., 2014. [Paper](https://arxiv.org/abs/1402.1892)  
   Motivation: F-measure-optimal decisions differ from a fixed 0.5 probability threshold. The implementation generalizes metric-aware selection to F0.5.

## Supporting libraries

- [Polars](https://docs.pola.rs/) for columnar, multithreaded Parquet processing.
- [RapidFuzz](https://rapidfuzz.github.io/RapidFuzz/) for optimized fuzzy string similarities.
- [scikit-learn](https://scikit-learn.org/) for TF-IDF and calibration utilities.
- [Transformers](https://huggingface.co/docs/transformers/) and [PyTorch](https://pytorch.org/) for multilingual fine-tuning and inference.

Always verify dependency and model licenses before redistribution. The E5 model cards identify their license; this package does not redistribute their weights.

