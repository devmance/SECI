"""
Variance decomposition — separates identity, model, and within-cell variance.

For each dimension, decompose the observed score variance across the
(model × identity) population into:

  - between-identity SD: spread of identity means (scaffold effect)
  - between-model SD:    spread of model means    (architecture effect)
  - within-cell SD:      residual at fixed (model, identity)

On the reference dataset TP has between-model SD 1.6× larger than
between-identity SD, locating TP variance primarily in model-architecture
differences. Reporting the decomposition every run makes this property
explicit on the per-dimension scores.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

from ..substrate.base import IdentitySubstrate

DIMENSIONS = ("ICT", "NCG", "PD", "TP", "CCC", "DEA")


def variance_decomposition(
    substrates: Iterable[IdentitySubstrate],
    arm: str = "A",
) -> Dict[str, Dict[str, float]]:
    """
    Decompose variance per dimension within a single arm.

    Returns:
        {dimension: {
            total_sd, between_identity_sd, between_model_sd,
            within_cell_sd, model_to_identity_ratio, verdict
        }, ...}
    """
    rows = [s for s in substrates if s.metadata.get("arm") == arm]
    if not rows:
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for d in DIMENSIONS:
        by_identity: Dict[str, List[float]] = defaultdict(list)
        by_model: Dict[str, List[float]] = defaultdict(list)
        by_cell: Dict[tuple, List[float]] = defaultdict(list)
        for s in rows:
            v = s.dimension_scores.get(d)
            if v is None:
                continue
            by_identity[s.metadata["identity"]].append(v)
            by_model[s.metadata["model"]].append(v)
            by_cell[(s.metadata["model"], s.metadata["identity"])].append(v)

        all_vals = np.array(
            [v for vs in by_cell.values() for v in vs], dtype=np.float64
        )
        if len(all_vals) < 2:
            continue

        identity_means = np.array(
            [np.mean(vs) for vs in by_identity.values()]
        )
        model_means = np.array(
            [np.mean(vs) for vs in by_model.values()]
        )

        within_cell_vars = [
            np.var(vs) for vs in by_cell.values() if len(vs) >= 2
        ]
        within_cell_sd = float(np.sqrt(np.mean(within_cell_vars))) if within_cell_vars else 0.0

        id_sd = float(identity_means.std())
        mod_sd = float(model_means.std())
        ratio = mod_sd / (id_sd + 1e-9)

        out[d] = {
            "total_sd": float(all_vals.std()),
            "between_identity_sd": id_sd,
            "between_model_sd": mod_sd,
            "within_cell_sd": within_cell_sd,
            "model_to_identity_ratio": ratio,
            "n_observations": int(len(all_vals)),
            "n_identities": int(len(by_identity)),
            "n_models": int(len(by_model)),
            "verdict": _classify_variance_decomp(ratio),
        }
    return out


def warning_flags(
    decomp: Dict[str, Dict[str, float]],
    portability: Dict[str, Dict[str, float]],
) -> List[str]:
    """
    Auto-generate human-readable warnings from a variance decomposition
    plus Claim-C portability table.
    """
    flags: List[str] = []
    for d, stats in decomp.items():
        if stats["model_to_identity_ratio"] > 1.3:
            flags.append(
                f"{d}: between-model SD ({stats['between_model_sd']:.2f}) exceeds "
                f"between-identity SD ({stats['between_identity_sd']:.2f}) at "
                f"{stats['model_to_identity_ratio']:.2f}× — variance on this "
                "dimension primarily reflects model-architecture differences rather "
                "than identity differences."
            )
    for d, stats in portability.items():
        if stats.get("mean_r", 0) < 0.1:
            flags.append(
                f"{d}: cross-model identity-ranking r = {stats['mean_r']:+.3f} "
                "(near zero) — identity rankings on this dimension do not replicate "
                "across model architectures."
            )
    return flags


def _classify_variance_decomp(ratio: float) -> str:
    if ratio > 1.5:
        return "MODEL DOMINATES — do not interpret as identity signal"
    if ratio > 1.1:
        return "model > identity — caution"
    if ratio > 0.7:
        return "comparable"
    if ratio > 0.3:
        return "identity > model"
    return "identity dominates"
