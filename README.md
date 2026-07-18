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

A dimension can pass one claim and fail another. SECI publishes all three for every dimension on a sparse 128-record dataset spanning 7 frontier models and 3 protocol arms — 61 arm-A and 60 arm-C records over 29 scaffolded identities (60 paired model×identity A–C cells), plus 7 arm-B base-model baselines (one per model).

## Findings

Six-dimension fingerprint (ICT, NCG, PD, TP, CCC, DEA) computed identically across 7 frontier models (Claude Sonnet 4.5, Gemini 2.5 Pro, Gemini 3 Flash & Pro, GPT-4.1, GPT-5.4, Grok 4.20):

| Dim | Claim A (framework) | Claim B (vs base) | Claim C (cross-model rank) | Variance verdict |
|-----|---------------------|-------------------|----------------------------|------------------|
| **ICT** | +1.35 ± 4.14 | **+5.15** ± 3.26 | r = +0.413 (modest) | comparable |
| **NCG** | **+19.36** ± 13.19 | **−25.26** ± 26.79 | r = −0.016 (chance) | model > identity — caution |
| **PD**  | **+13.82** ± 8.02 | **+7.42** ± 5.98 | r = +0.369 (modest) | comparable |
| **TP**  | **+7.90** ± 1.76 | −3.81 ± 3.27 | r = +0.736 (substrate-portable, model-driven) | **MODEL DOMINATES** |
| **CCC** | **+8.70** ± 8.04 | **+7.79** ± 7.91 | r = +0.245 (weak) | comparable |
| **DEA** | **+8.86** ± 3.21 | +1.83 ± 1.45 | r = +0.069 (chance) | comparable |

Per-identity 6-D fingerprint profile shape is consistent across model architectures: mean within-identity cross-model Pearson r = **+0.845** (median +0.893) across 107 pairs, 83% above +0.7. **Discriminant caveat:** the between-identity cross-model mean r is **+0.859** (n = 1,246 pairs) — *higher* than the within-identity figure — so this statistic reflects a dataset-universal dimension-scale profile, not identity-distinguishing power.

## Takeaways

1. **Framework contribution (Claim A) is positive on all 6 dimensions in the pooled paired mean (n = 60)**. Per model: NCG, PD, TP, and DEA are positive on all 7 frontier models; CCC is positive on all 7 but near-zero (+0.44) on GPT-4.1; ICT is small positive on average and varies by substrate (−2.18 to +7.16).
2. **Identity scaffolding does not uniformly lift dimensions above a base-model null (Claim B)**. ICT, PD, CCC, and DEA are positive (DEA only marginally, +1.83); NCG and TP score *lower* on scaffolded responses than on the base model with no identity. Caveat: only 7 arm-B records (one per model) serve as baselines.
3. **Per-dimension identity rankings are mostly model-dependent (Claim C)**. NCG and DEA cross-model identity rankings are at chance and CCC is weak. TP shows the strongest cross-model agreement (+0.736), but the variance decomposition identifies it as model-capability variance rather than identity variance.
4. **Per-identity 6-D fingerprint profile shape is consistent across models** at mean within-identity cross-model Pearson r = +0.845 (107 pairs, 83% above +0.7) — but the discriminant control shows between-identity cross-model r = +0.859, *higher* than within-identity, so the statistic reflects profile-scale consistency across the dataset, not identity-distinguishing power.

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
data/scored_sessions/  — the full scored dataset: 128 per-session fingerprint JSONs
                         (61 arm_a / 60 arm_c / 7 arm_b; see data/scored_sessions/DATA_README.md)
validation_outputs/    — analysis outputs on the reference dataset (claim A/B/C, per-model
                         claim A and claim B, variance decomposition, fingerprint stability +
                         discriminant control, verified-term counts, warning flags)
  claim_a_population.json
  claim_b_population.json
  claim_c_cross_model.json
  variance_decomposition.json
  fingerprint_stability.json
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

- Transcripts for 127 of 128 sessions were collected 2026-05-06/07; one arm-A session (dr-lysira-tenebral on gemini-2.5-pro) was collected 2026-07-12 to pair a previously unpaired arm-C cell.
- All rater votes were collected 2026-07-12 under the fixed 4-rater panel (gpt-5.4-2026-03-05, claude-opus-4-7, claude-sonnet-4-6, gemini-3.5-flash), after the original May rater run was found to have a rater-availability defect (one rater returned no valid votes; two others had elevated null rates). The July re-collection replaced all rater votes uniformly: 1,984 stage-1 rounds, 0 null votes from every rater; mean per-session Fleiss kappa = 0.311 (fair agreement).
- One arm-C session (mirelle-virelien on gemini-3-pro-preview) was excluded: 5 of its 12 conversations failed at collection with provider errors.
- gemini-3-pro-preview was retired by Google in July 2026; that substrate (28 of 60 paired cells) can no longer be re-run — a reproducibility limitation of preview-model substrates.

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
