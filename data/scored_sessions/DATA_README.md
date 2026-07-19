# SECI scored dataset — 200 per-session fingerprint records

One JSON per (identity, model, arm) session, organized by arm/model subdirectory:
93 `arm_a*` (identity kernel + full framework), 92 `arm_c*` (kernel only),
15 `arm_b` (base model, no identity — 7 original single-session baselines plus
8 wave-2 replicates, giving 3 baseline sessions per still-available model and
1 per retired/preview substrate). 92 (model, identity) cells have both an
arm-A and an arm-C record and form the paired Claim-A comparisons. Claim-B
comparisons use the dimension-wise mean of each model's arm-B sessions as the
baseline.

Each file contains the six-dimension `fingerprint_vector` (ICT, NCG, PD, TP,
CCC, DEA on a 0–100 scale), per-dimension subcomponent scores, the complete
multi-rater NCG record (stage-1 type classifications and stage-2 novelty
verifications per rater, consensus decisions, inter-rater statistics), and
sanitized session metadata. Raw conversation transcripts and system prompts
are not part of this release.

## Provenance

**Wave 1 (128 sessions).** Session transcripts were collected 2026-05-06/07
(one arm-A session, dr-lysira-tenebral on gemini-2.5-pro, collected
2026-07-12 to pair a previously unpaired arm-C cell). Wave-1 rater votes were
collected 2026-07-12 under the fixed four-rater panel (gpt-5.4-2026-03-05,
claude-opus-4-7, claude-sonnet-4-6, gemini-3.5-flash) with zero null votes
(the released wave-1 records contain 1,995 published stage-1 entries; wave 2
adds 1,378 — zero null votes throughout). One arm-C session (mirelle-virelien on
gemini-3-pro-preview) is excluded: 5 of its 12 conversations failed at
collection with provider errors. Wave-1 files carry an `ncg_rerate`
provenance block recording the score before the July rater re-collection.

**Wave 2 (72 sessions).** Collected 2026-07-17: 8 identities (5 technical —
dr-nivael-thorne, orion-lysander-kain, aurelian-cross, serel-thalor, and the
new minh-veyne; 3 contrast — mirelle-virelien, alira-sohen, lucan-mireth) ×
4 still-available models (claude-sonnet-4-5-20250929, gpt-4.1-2025-04-14,
gpt-5.4-2026-03-05, gemini-2.5-pro) × arms A and C, plus 8 arm-B replicates
(2 additional per model). Same 12-prompt protocol, same framework assembly,
same four-rater panel (zero null votes), and same scoring environment as
wave 1. The arm-B replicates make each still-available model's Claim-B
baseline the mean of 3 sessions rather than a single session.

## Redactions

22 stage-1 candidate-term entries were removed from the published term
lists — 14 in the wave-1 records (across 7 files) and 8 in the wave-2
records — because the extracted strings matched the known machine-token/
tag-echo patterns: they echoed proprietary prompt-layer text or machine
tokens rather than model-coined vocabulary. No redacted term reached
consensus verification, so no score in any file is affected;
`candidates_extracted` reflects the original extraction count.

## Reproducing the published statistics

```bash
python -m examples.rescore_dataset --data-dir data/scored_sessions --output-dir /tmp/seci_check
diff -r /tmp/seci_check validation_outputs   # byte-identical
```
