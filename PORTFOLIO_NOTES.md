# Portfolio use

Recommended repository title: **Multilingual Business Entity Resolution — 0.98076 macro F0.5**

Recommended résumé bullet:

> Built a multilingual entity-resolution pipeline for 1.73M query businesses against 9.97M candidates, raising public macro F0.5 from 0.929 to 0.98076 using hybrid sparse/dense blocking, LightGBM, multilingual cross-encoder reranking, graph context, and metric-aware decoding.

Recommended interview structure:

1. Explain why blocking recall caps every later model.
2. Show how pair features and cross-encoder evidence complement each other.
3. Explain why competition, rank, and ownership require Stage 2.
4. Explain F0.5-aware decoding and why a fixed 0.5 threshold is unsuitable.
5. Discuss rejected micro/fine-tuning experiments as evidence of controlled evaluation.

Before publishing:

- Confirm every team member's attribution.
- Add a project license approved by all owners.
- Confirm competition rules permit publishing source and the final prediction file.
- Prefer the source ZIP for GitHub. Store the complete ZIP with TSV in a release or large-file store.
- Never commit dataset files, downloaded checkpoints, credentials, or server caches.
