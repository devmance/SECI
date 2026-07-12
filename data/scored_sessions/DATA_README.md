# SECI scored dataset — 128 per-session fingerprint records

One JSON per (identity, model, arm) session, organized by arm/model subdirectory:
61 `arm_a*` (identity kernel + full framework), 60 `arm_c*` (kernel only),
7 `arm_b` (base model, no identity — one per model). 60 (model, identity) cells
have both an arm-A and an arm-C record and form the paired Claim-A comparisons.

Each file contains the six-dimension `fingerprint_vector` (ICT, NCG, PD, TP,
CCC, DEA on a 0–100 scale), per-dimension subcomponent scores, the complete
multi-rater NCG record (stage-1 type classifications and stage-2 novelty
verifications per rater, consensus decisions, inter-rater statistics), and
sanitized session metadata. Raw conversation transcripts and system prompts
are not part of this release.

## Provenance

Session transcripts were collected 2026-05-06/07 (one arm-A session,
dr-lysira-tenebral on gemini-2.5-pro, collected 2026-07-12 to pair a
previously unpaired arm-C cell). All rater votes were collected 2026-07-12
under the fixed four-rater panel (gpt-5.4-2026-03-05, claude-opus-4-7,
claude-sonnet-4-6, gemini-3.5-flash) with zero null votes across 1,984
stage-1 rounds. One arm-C session (mirelle-virelien on gemini-3-pro-preview)
is excluded: 5 of its 12 conversations failed at collection with provider
errors. Files carry an `ncg_rerate` provenance block recording the score
before the July rater re-collection.

## Redactions

14 of 3,968 stage-1 candidate-term entries (across 7 files) were removed from
the published term lists because the extracted strings echoed proprietary
prompt-layer text rather than model-coined vocabulary. No redacted term
reached consensus verification, so no score in any file is affected;
`candidates_extracted` reflects the original extraction count.

## Reproducing the published statistics

```bash
python -m examples.rescore_dataset --data-dir data/scored_sessions --output-dir /tmp/seci_check
diff -r /tmp/seci_check validation_outputs   # byte-identical
```
