"""
SECI scoring layer.

Two entry points:
  - DeterministicScorer (alias: SECIAnalyzer) — the embedding/regex/compression
    pipeline that scores ICT, PD, TP, CCC, DEA on the raw conversation
    responses. NCG falls back to the deterministic neologism + semantic-novelty
    sub-scorers when no LLM raters are configured.
  - SECIScorer — extends DeterministicScorer with multi-rater NCG verification
    (≥3-of-4 frontier-LLM consensus, Fleiss kappa, pairwise Cohen kappa).

Both produce a six-dimension fingerprint vector (ICT, NCG, PD, TP, CCC, DEA)
on a 0-100 scale plus per-dimension provenance in dimension_details.
"""

from .dimensions import SECIAnalyzer
from .analyzer import SECIScorer, DeterministicScorer

__all__ = ["SECIAnalyzer", "SECIScorer", "DeterministicScorer"]
