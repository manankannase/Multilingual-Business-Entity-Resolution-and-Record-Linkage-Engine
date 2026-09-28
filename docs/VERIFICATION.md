# Package verification

Packaging checks completed on 28 September 2026:

- All included Python files compiled with `compileall`.
- Final TSV header matched `source1_entity_id` and `matched_entity_ids`.
- Final TSV contained 1,732,544 data rows and 1,732,544 unique Source-1 IDs.
- Final TSV contained 97,702 empty-match rows and zero duplicate Source-1 IDs.
- Final TSV SHA-256 matched the server-produced artifact.
- Fine-tuning CPU integration test passed: preparation/resume, fold isolation, score coverage, entity evaluation, rejection gate, and input-drift protection.
- Micro V2 unit tests passed for feature and balanced-partition behavior.
- Micro V2 integration test passed: preflight, training, calibration, comparison, five TSVs, unique ownership, baseline hash preservation, and overwrite guards.

GPU-only tests requiring PyTorch/model weights were not executed during Mac packaging. They had passed the recorded server runtime preflight before the A40 fine-tuning run. The official competition validator cannot be rerun without the licensed dataset and generated candidate file; its recorded production result was PASS.
