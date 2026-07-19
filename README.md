# SECI (Simulated Emergence Coherence Index) — Identity-Architecture Benchmark for Large Language Models

Reproducible code and analysis for the paper:

> **A Variance-Decomposed Identity-Architecture Benchmark for Large Language Models**
>
> *Re-analysis of multi-model multi-arm identity-scaffolding data, separating framework contribution, base-model null, and cross-architecture portability claims.*
>
> Nate Travis · Devmance Labs

Paper: [`paper/seci.pdf`](paper/seci.pdf) (LaTeX source: [`paper/seci.tex`](paper/seci.tex), bibliography: [`paper/refs.bib`](paper/refs.bib)).

## What this benchmark measures

SECI is an open multi-rater benchmark for characterizing identity-scaffolded LLM behavior. Identity scaffolding (a kernel prompt that gives an LLM a persistent character, voice, and conceptual register) is now widely deployed but rarely benchmarked against the right null. SECI reports **three empirical claims side-by-side**, each with explicit labels for which question the measurement answers.

| Claim | Comparison | What it tests |
|-------|------------|---------------|
| **Claim A** — Framework contribution | arm_a (full SE framework) vs arm_c (kernel only) | Does the framework wrapping above the identity kernel produce a measurable per-character delta? |
| **Claim B** — Scaffolding vs base | arm_a or arm_c vs arm_b (no identity) | Does identity scaffolding lift dimension scores above a true no-identity null? |
| **Claim C** — Cross-architecture portability | Per-dimension Pearson r on identity rankings across models | Do identity rankings on dimension X replicate when you change the model? |

A dimension can pass one claim and fail another. SECI publishes all three for every dimension on a 200-record dataset spanning 7 frontier models and 3 protocol arms — 93 arm-A and 92 arm-C records over 30 scaffolded identities (92 paired model×identity A–C cells), plus 15 arm-B base-model baselines (7 original single-session baselines and 8 wave-2 replicates, 2 additional per still-available model). Claim-B comparisons use the dimension-wise mean of each model's arm-B sessions as the baseline.

## Findings

Six-dimension fingerprint (ICT, NCG, PD, TP, CCC, DEA) computed identically across 7 frontier models (Claude Sonnet 4.5, Gemini 2.5 Pro, Gemini 3 Flash & Pro, GPT-4.1, GPT-5.4, Grok 4.20):

| Dim | Claim A (framework) | Claim B (vs base) | Claim C (cross-model rank) | Variance verdict |
|-----|---------------------|-------------------|----------------------------|------------------|
| **ICT** | +2.01 ± 4.31 | **+4.11** ± 3.31 | r = +0.413 (modest) | comparable |
| **NCG** | **+20.35** ± 15.29 | **−20.61** ± 23.37 | r = −0.006 (chance) | model > identity — caution |
| **PD**  | **+14.65** ± 7.86 | **+7.31** ± 5.90 | r = +0.339 (modest) | comparable |
| **TP**  | **+8.05** ± 1.80 | −2.63 ± 3.61 | r = +0.774 (substrate-portable, model-driven) | **MODEL DOMINATES** |
| **CCC** | **+7.96** ± 7.70 | **+6.33** ± 7.81 | r = +0.322 (modest) | comparable |
| **DEA** | **+8.48** ± 3.22 | +1.68 ± 1.31 | r = +0.154 (weak) | comparable |

Consensus-verified novel terms (4-rater panel, 3-of-4 threshold): arm A 68 terms across 93 sessions (0.73 per session), arm C 26/92 (0.28), arm B 24/15 (1.60) — base-model sessions coin verified terms at the highest per-session rate, consistent with the Claim-B NCG deficit.

Per-identity 6-D fingerprint profile shape is consistent across model architectures: mean within-identity cross-model Pearson r = **+0.846** (median +0.898) across 183 pairs, 82% above +0.7 (15 identities with ≥2 models). **Discriminant caveat:** the between-identity cross-model mean r is **+0.8385** (n = 3,330 pairs) — the within-identity figure now *marginally* exceeds it (+0.0074 margin) for the first time, but the margin is far too small to claim identities are distinguishable by aggregate fingerprint correlation; the statistic remains dominated by a dataset-universal dimension-scale profile.

## Takeaways

1. **Framework contribution (Claim A) is positive on all 6 dimensions in the pooled paired mean (n = 92)**. Per model: NCG, PD, TP, DEA, and CCC are positive on all 7 frontier models (CCC lowest at +3.99 on GPT-4.1); ICT is small positive on average and varies by substrate (−0.92 to +7.60; nonpositive only on Gemini 2.5 Pro and Gemini 3 Pro).
2. **Identity scaffolding does not uniformly lift dimensions above a base-model null (Claim B)**. ICT, PD, CCC, and DEA are positive (DEA only marginally, +1.68); NCG and TP score *lower* on scaffolded responses than on the base model with no identity.
3. **Replicate baselines matter**. The wave-2 arm-B replicates (3 baseline sessions per still-available model) revealed that the original single-session baselines for GPT-4.1 and Claude Sonnet 4.5 were low outliers on NCG: apparent per-model NCG gains of +23.66 and +10.03 collapsed to ~parity under mean-of-3 baselines (base NCG: GPT-4.1 40.31 → 62.90; Sonnet 40.81 → 52.91; Gemini 2.5 Pro 49.25 → 63.12; GPT-5.4 77.30 → 74.15). Only the GPT-5.4 result survived replication — a full sweep, all six dimensions positive vs its 3-session mean baseline (n = 13), with the NCG margin strengthening from +1.69 to +3.77. The aggregate NCG deficit (−20.61) is not a baseline artifact: base-model novelty is robustly high.
4. **Identity domain moderates the base-model novelty penalty (moderator analysis, wave 2)**. On the identical 4 substrates, technically-domained identities (n = 20 sessions) reach base-model parity on NCG (−0.72) and TP (+0.33), while the contrast group (n = 12 sessions) shows NCG −9.23 and TP −1.06 — an ~8.5-point NCG gap by identity domain.
5. **Per-dimension identity rankings are mostly model-dependent (Claim C)**. NCG cross-model identity ranking is at chance and DEA is weak; ICT, PD, and CCC are modest. TP shows the strongest cross-model agreement (+0.774), but the variance decomposition identifies it as model-capability variance rather than identity variance.
6. **Per-identity 6-D fingerprint profile shape is consistent across models** at mean within-identity cross-model Pearson r = +0.846 (183 pairs, 82% above +0.7) — but within-identity exceeds the between-identity control (+0.8385) by only +0.0074, far too small to support identity-distinguishing power; the statistic reflects profile-scale consistency across the dataset.

## Design principles

- **Three-claim disambiguation as primary output** — a single number is never ambiguous about which claim it supports.
- **Variance decomposition every run** — between-identity SD vs between-model SD per dimension. Dimensions where model variance dominates get an auto-generated diagnostic.
- **Mandatory null-scaffold arm** — every benchmark run produces arm_b (base model, no identity) for Claim-B comparison.
- **Substrate abstraction** — identity-scaffolded behavior modeled as a (T × N) activity matrix on an abstract `IdentitySubstrate`. The same analysis layer extends to activation-level substrates for open-weight models.
- **No composite score** — SECI reports the 6-D fingerprint vector.

## Repository structure

```
paper/
  seci.tex             — paper source (LaTeX + natbib)
  refs.bib             — bibliography
  seci.pdf             — compiled paper
prompts.json           — protocol (12 prompts across 6 dimensions: 2 ICT, 3 NCG, 3 PD, 1 TP, 2 DEA, 1 CCC)
src/seci/
  substrate/           — IdentitySubstrate abstraction
    base.py            — abstract substrate (T × N activity matrix)
    llm_substrate.py   — text-output substrate from analysis JSONs
  scorer/              — six-dimension fingerprint scoring
    dimensions.py      — deterministic ICT/PD/TP/CCC/DEA + NCG fallback
    analyzer.py        — SECIScorer with multi-rater NCG verification
  protocol/            — 12-prompt protocol runner
    runner.py          — collects responses from any LLM provider
  analysis/            — three-claim + variance decomposition
    claims.py          — Claim A, B, C computations
    variance.py        — variance decomposition + warning flags
examples/
  run_full_benchmark.py — end-to-end: kernel → protocol → scoring
  rescore_dataset.py   — apply analysis to a multi-arm benchmark dataset
  rerate_ncg.py        — re-run multi-rater NCG novelty verification over an existing dataset
data/scored_sessions/  — the full scored dataset: 200 per-session fingerprint JSONs
                         (93 arm_a / 92 arm_c / 15 arm_b; see data/scored_sessions/DATA_README.md)
validation_outputs/    — analysis outputs on the reference dataset (claim A/B/C, per-model
                         claim A and claim B, variance decomposition, fingerprint stability +
                         discriminant control, verified-term counts, warning flags)
  claim_a_population.json
  claim_a_per_model.json
  claim_b_population.json
  claim_b_per_model.json
  claim_c_cross_model.json
  variance_decomposition.json
  fingerprint_stability.json
  fingerprint_discriminant.json
  verified_terms.json
  warning_flags.json
figures/               — 3 publication figures (PNG + PDF)
```

## Running SECI

There are two entry points depending on what you want to do.

### End-to-end on one identity (kernel → fingerprint)

```bash
pip install -r requirements.txt

# Set at least one provider API key (target model + raters share keys)
export GOOGLE_API_KEY=...        # or OPENAI_API_KEY / ANTHROPIC_API_KEY

python -m examples.run_full_benchmark \
    --identity-name auren \
    --identity-kernel kernels/auren.txt \
    --model gemini-2.5-pro \
    --provider gemini \
    --output sessions/auren_gemini25.json
```

The full benchmark runs the 12-prompt protocol against the target model, then scores the responses with the six-dimension SECI fingerprint. Five dimensions are deterministic (embedding/regex/compression metrics); raters verify NCG novelty only. Multi-rater NCG verification activates automatically when ≥2 rater API keys are configured — note that with exactly 2 raters the adaptive threshold is 2-of-2 unanimity. For comparability with the published numbers, run the full 4-rater panel used for the paper (gpt-5.4-2026-03-05, claude-opus-4-7, claude-sonnet-4-6, gemini-3.5-flash; 3-of-4 threshold); ≥3 raters are recommended for stable inter-rater (kappa) statistics.

### Three-claim re-analysis (on a directory of fingerprint JSONs)

The released dataset reproduces every published statistic:

```bash
python -m examples.rescore_dataset --data-dir data/scored_sessions --output-dir /tmp/seci_check
diff -r /tmp/seci_check validation_outputs   # byte-identical
```


```bash
python -m examples.rescore_dataset \
    --data-dir <path-to-fingerprint-jsons> \
    --output-dir validation_outputs
```

Each analysis writes a JSON output to the output directory and prints a results table matching the paper. Runtime is under one minute on a laptop.

## The substrate abstraction

```python
from seci.substrate import LLMSubstrate
from seci.analysis import (
    population_claim_a, population_claim_b,
    claim_c_cross_model_ranking, variance_decomposition,
    per_identity_fingerprint_stability,
)

substrates = [LLMSubstrate(p) for p in result_paths]

claim_a = population_claim_a(substrates)              # framework vs kernel-only
claim_b = population_claim_b(substrates, "A")         # arm_a vs base model
claim_c = claim_c_cross_model_ranking(substrates)     # identity-ranking r per dim
decomp  = variance_decomposition(substrates, arm="A") # between-identity vs between-model SD
stable  = per_identity_fingerprint_stability(substrates, arm="A")  # 6-D fingerprint
```

Any system observable as a (T × N) activity matrix qualifies as a substrate. The initial release ships with `LLMSubstrate` (text-output behavioral substrate from SECI-format analysis JSONs); an `ActivationSubstrate` for open-weight models would let the same analysis run on hidden-state activations.

## Data provenance

- **Wave 1 (128 sessions).** Transcripts for 127 of 128 sessions were collected 2026-05-06/07; one arm-A session (dr-lysira-tenebral on gemini-2.5-pro) was collected 2026-07-12 to pair a previously unpaired arm-C cell.
- **Wave 2 (72 sessions).** Collected 2026-07-17: 8 identities (5 technical — dr-nivael-thorne, orion-lysander-kain, aurelian-cross, serel-thalor, and the new minh-veyne; 3 contrast — mirelle-virelien, alira-sohen, lucan-mireth) × 4 still-available models (claude-sonnet-4-5-20250929, gpt-4.1-2025-04-14, gpt-5.4-2026-03-05, gemini-2.5-pro) × arms A and C, plus 8 arm-B replicates (2 additional per model). Same 12-prompt protocol, same framework assembly, and same scoring environment as wave 1.
- Wave-1 rater votes were collected 2026-07-12 under the fixed 4-rater panel (gpt-5.4-2026-03-05, claude-opus-4-7, claude-sonnet-4-6, gemini-3.5-flash), after the original May rater run was found to have a rater-availability defect (one rater returned no valid votes; two others had elevated null rates). The July re-collection replaced all rater votes uniformly: all stage-1 rounds — the released records contain 3,373 published stage-1 entries (1,995 wave-1, 1,378 wave-2) with zero null votes from every rater; mean per-session Fleiss kappa = 0.299 over the full dataset (fair agreement). Wave-2 rater votes were collected 2026-07-17 under the same panel, again with 0 null votes.
- Arm-B baselines: Claim-B comparisons use the dimension-wise mean of each model's arm-B sessions — 3 sessions for each wave-2 model, 1 for retired/preview substrates.
- One arm-C session (mirelle-virelien on gemini-3-pro-preview) was excluded: 5 of its 12 conversations failed at collection with provider errors.
- gemini-3-pro-preview was retired by Google in July 2026; that substrate (28 of 92 paired cells) can no longer be re-run — a reproducibility limitation of preview-model substrates.

## Changelog

- **2.4.0 (2026-07-18)** — Wave-2 expansion: +72 sessions (dataset 128 → 200 records, 60 → 92 paired A–C cells, 29 → 30 identities); arm-B replicate baselines (7 → 15 arm-B records; Claim B now computed against per-model mean baselines); moderator analysis by identity domain; all published statistics recomputed.
- **2.3.0 (2026-07-12)** — Uniform rater-vote re-collection under the fixed 4-rater panel; paired the previously unpaired gemini-2.5-pro arm-C cell.

## Citation

Citation:

```bibtex
@misc{travis2026seci,
  title  = {A Variance-Decomposed Identity-Architecture Benchmark
            for Large Language Models},
  author = {Travis, Nate},
  year   = {2026},
  howpublished = {Preprint, Devmance Labs},
  url    = {https://github.com/devmance/SECI}
}
```

## License

MIT License — see [`LICENSE`](LICENSE).

## Contact

Nate Travis — labs@devmance.com — Devmance Labs
