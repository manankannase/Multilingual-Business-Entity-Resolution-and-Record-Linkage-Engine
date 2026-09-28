# TEJAS_v3 versus Edited_version

The extracted code directories have identical configuration, blocking, feature
extraction, context, cross-encoder, dense retrieval, decision, metric, data,
normalization and Slurm files. Only stage2.py and error_analysis.py differ.

TEJAS_v3 adds stage-1 fold-model persistence; simulated removal of 20% of Source-1
owners; recomputation of reverse ranks and stage-1 probabilities; two augmented
training copies; validation on a removed-owner subset; and an error-analysis
reader for the subset's entity IDs.

No evaluated submission or validation result for that modification is included
in this ZIP. The included model_v1 artifacts are Stage 1, not proof of an improved
Stage-2 score. Its comments assume train/test orphan percentages that are not
established by the supplied artifacts. Hidden test identities are unavailable.

Recomputing stage-1 probabilities can admit pairs outside the cached XENC
shortlist; its code tolerates missing XENC scores. The simulated reverse ranks
are also computed from the available candidate graph, not a rerun of original
retrieval against the remaining Source-1 corpus. These are approximations.

Our enhancement retains original features, ranks and probabilities and treats
removing owners solely as context augmentation. It includes original training
context, uses 10% and 20% removal conditions, requires complete XENC coverage,
and compares new predictions on a separate validation group and a shared stress
graph. Stage groups are preferred; normalized-name groups are used when too few
states exist. This does not solve retrieval misses or claim to reconstruct test.

Local audit: among scored validation entities there are 1,039 retrieval misses,
581 retrieved true pairs absent from Stage-2 scores, and 6,905 scored true pairs
not selected. This motivates testing competition context, while preserving
false-positive control. Those pair counts do not weight entities equally.

The acceptance check is conservative and may reject the experiment. Public
leaderboard improvement remains unverified until a generated submission is
evaluated. CPU training and output validation must run with the server caches.
