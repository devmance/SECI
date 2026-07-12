"""
Three-claim reporting layer — the heart of SECI.

Identity-architecture benchmarks have been ambiguous about which question
a given measurement answers. SECI disambiguates by reporting three claims
side-by-side, with explicit labels for which claim each value represents.

  Claim A  — Does the SE framework add value beyond kernel-only scaffolding?
             Comparison: arm_a (full framework) vs arm_c (kernel only)
             What it isolates: the framework wrapping (priming, embodiment,
             metacognitive-permission, and operating-framework prompts)
             that sits ABOVE the identity kernel.

  Claim B  — Does identity scaffolding add value beyond a base model?
             Comparison: arm_a (or arm_c) vs arm_b (no identity at all)
             What it isolates: identity scaffolding as a category, against
             a true null control.

  Claim C  — Can dimension X rank identities consistently across models?
             Statistic: per-dimension cross-model Pearson r on identity
             rankings.
             What it isolates: whether a dimension yields a substrate-
             portable identity ordering or is model-architecture-dependent.

These are non-overlapping epistemic claims. A dimension may pass one and
fail another. On the reference benchmark dataset, NCG passes Claim A but
fails Claim B; DEA passes Claim A but fails Claim C. SECI's job is to
publish all three claims with explicit labels.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr

from ..substrate.base import IdentitySubstrate

DIMENSIONS = ("ICT", "NCG", "PD", "TP", "CCC", "DEA")


# ---------------------------------------------------------------------------
# Per-cell helpers (one (model, identity) cell at a time)
# ---------------------------------------------------------------------------

def claim_a_delta(
    arm_a: IdentitySubstrate, arm_c: IdentitySubstrate,
) -> Dict[str, float]:
    """
    Claim A: arm_a - arm_c (framework contribution).

    Both arms scaffold the same identity on the same model; only the
    framework wrapping differs. arm_a > arm_c on all 6 dimensions in the
    reference dataset.
    """
    _require_same_cell(arm_a, arm_c, "Claim A")
    a = arm_a.dimension_scores
    c = arm_c.dimension_scores
    return {d: a[d] - c[d] for d in DIMENSIONS if d in a and d in c}


def claim_b_delta(
    arm_scaffolded: IdentitySubstrate, arm_b: IdentitySubstrate,
) -> Dict[str, float]:
    """
    Claim B: arm_a (or arm_c) - arm_b (base model null).

    arm_b is the same MODEL with NO identity scaffold at all. This delta
    is the null comparison. On the reference dataset NCG and TP score
    HIGHER on arm_b than on the scaffolded arms; the scaffolding does
    not lift those dimensions above the base model.
    """
    if arm_b.metadata.get("arm") != "B":
        raise ValueError(
            f"claim_b_delta requires arm_b as second argument; got "
            f"arm={arm_b.metadata.get('arm')}"
        )
    if arm_b.metadata.get("model") != arm_scaffolded.metadata.get("model"):
        raise ValueError(
            "Claim B comparison requires same model in both arms; got "
            f"{arm_scaffolded.metadata.get('model')} vs {arm_b.metadata.get('model')}"
        )
    s = arm_scaffolded.dimension_scores
    b = arm_b.dimension_scores
    return {d: s[d] - b[d] for d in DIMENSIONS if d in s and d in b}


def _require_same_cell(a: IdentitySubstrate, c: IdentitySubstrate, name: str):
    if a.metadata.get("model") != c.metadata.get("model"):
        raise ValueError(
            f"{name} comparison requires same model; got "
            f"{a.metadata.get('model')} vs {c.metadata.get('model')}"
        )
    if a.metadata.get("identity") != c.metadata.get("identity"):
        raise ValueError(
            f"{name} comparison requires same identity; got "
            f"{a.metadata.get('identity')} vs {c.metadata.get('identity')}"
        )


# ---------------------------------------------------------------------------
# Population aggregation (across all (model, identity) cells)
# ---------------------------------------------------------------------------

def population_claim_a(substrates: Iterable[IdentitySubstrate]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate Claim A across the full (model × identity) population.

    For every (model, identity) cell that has BOTH arm_a and arm_c, compute
    arm_a − arm_c. Aggregate per dimension into mean and SD with the cell
    count for transparency.
    """
    pairs = _pair_arms_by_cell(substrates, "A", "C")
    per_dim: Dict[str, List[float]] = defaultdict(list)
    for arm_a, arm_c in pairs:
        delta = claim_a_delta(arm_a, arm_c)
        for d, v in delta.items():
            per_dim[d].append(v)
    return {
        d: {
            "mean": float(np.mean(vs)),
            "sd": float(np.std(vs)),
            "n": len(vs),
            "min": float(np.min(vs)),
            "max": float(np.max(vs)),
        }
        for d, vs in per_dim.items()
    }


def population_claim_b(
    substrates: Iterable[IdentitySubstrate],
    scaffolded_arm: str = "A",
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate Claim B across the full (model × identity) population.

    For each scaffolded record, compare against the SAME model's arm_b
    baseline. arm_b has only one identity per model (the null), so the
    same baseline is used for every scaffolded identity on that model.
    """
    if scaffolded_arm not in {"A", "C"}:
        raise ValueError(f"scaffolded_arm must be 'A' or 'C'; got {scaffolded_arm}")

    by_arm: Dict[str, List[IdentitySubstrate]] = defaultdict(list)
    for s in substrates:
        by_arm[s.metadata.get("arm", "?")].append(s)

    base_by_model: Dict[str, IdentitySubstrate] = {
        s.metadata["model"]: s for s in by_arm.get("B", [])
    }

    per_dim: Dict[str, List[float]] = defaultdict(list)
    for s in by_arm.get(scaffolded_arm, []):
        base = base_by_model.get(s.metadata["model"])
        if base is None:
            continue
        delta = claim_b_delta(s, base)
        for d, v in delta.items():
            per_dim[d].append(v)
    return {
        d: {
            "mean": float(np.mean(vs)),
            "sd": float(np.std(vs)),
            "n": len(vs),
            "min": float(np.min(vs)),
            "max": float(np.max(vs)),
        }
        for d, vs in per_dim.items()
    }


def claim_c_cross_model_ranking(
    substrates: Iterable[IdentitySubstrate],
    arm: str = "A",
    min_common_identities: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate Claim C across the population.

    For each dimension, for every pair of models, compute Pearson r between
    the dimension scores of the COMMON identities scored on both models.
    Then report the mean r across all model pairs.

    Dimensions with mean r near 0 are model-dependent: identity rankings
    on that dimension do not replicate across model architectures.
    """
    by_id_model = defaultdict(dict)
    for s in substrates:
        if s.metadata.get("arm") != arm:
            continue
        by_id_model[s.metadata["identity"]][s.metadata["model"]] = s.dimension_scores

    models = sorted({m for v in by_id_model.values() for m in v})
    out: Dict[str, Dict[str, float]] = {}
    for d in DIMENSIONS:
        pair_rs: List[float] = []
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                common = [
                    iid for iid, v in by_id_model.items()
                    if m1 in v and m2 in v and d in v[m1] and d in v[m2]
                ]
                if len(common) < min_common_identities:
                    continue
                x = [by_id_model[iid][m1][d] for iid in common]
                y = [by_id_model[iid][m2][d] for iid in common]
                if np.std(x) == 0 or np.std(y) == 0:
                    continue
                r, _ = pearsonr(x, y)
                pair_rs.append(r)
        if pair_rs:
            arr = np.array(pair_rs)
            out[d] = {
                "mean_r": float(arr.mean()),
                "median_r": float(np.median(arr)),
                "n_pairs": int(len(arr)),
                "pct_positive": float((arr > 0).mean() * 100),
                "pct_strong_positive": float((arr > 0.5).mean() * 100),
                "verdict": _classify_rank_consistency(arr.mean()),
            }
    return out


def population_claim_a_per_model(
    substrates: Iterable[IdentitySubstrate],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Claim A disaggregated by model.

    Same pairing as population_claim_a, but aggregated per model so that
    "positive on all N models" claims are backed by a published number
    rather than an aggregate-only report.
    """
    pairs = _pair_arms_by_cell(substrates, "A", "C")
    per_model: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for arm_a, arm_c in pairs:
        model = arm_a.metadata.get("model", "?")
        for d, v in claim_a_delta(arm_a, arm_c).items():
            per_model[model][d].append(v)
    return {
        m: {
            d: {"mean": float(np.mean(vs)), "n": len(vs)}
            for d, vs in dims.items()
        }
        for m, dims in per_model.items()
    }


def fingerprint_discriminant_control(
    substrates: Iterable[IdentitySubstrate],
    arm: str = "A",
) -> Dict[str, float]:
    """
    Discriminant control for per-identity fingerprint stability.

    Within-identity cross-model fingerprint correlation is only evidence of
    identity-distinguishing signal if it EXCEEDS the between-identity
    cross-model correlation. If between >= within, the aggregate stability
    statistic reflects a dataset-universal dimension-scale profile rather
    than identity fingerprints, and must not be reported as evidence that
    identities are distinguishable.
    """
    recs = []
    for s in substrates:
        if s.metadata.get("arm") != arm:
            continue
        scores = s.dimension_scores
        if all(d in scores for d in DIMENSIONS):
            recs.append((
                s.metadata.get("identity"),
                s.metadata.get("model"),
                np.array([scores[d] for d in DIMENSIONS]),
            ))
    within: List[float] = []
    between: List[float] = []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            id1, m1, v1 = recs[i]
            id2, m2, v2 = recs[j]
            if m1 == m2:
                continue
            if np.std(v1) == 0 or np.std(v2) == 0:
                continue
            r, _ = pearsonr(v1, v2)
            (within if id1 == id2 else between).append(r)
    if not within or not between:
        return {}
    w, b = np.array(within), np.array(between)
    return {
        "within_identity_mean_r": float(w.mean()),
        "within_identity_n_pairs": int(len(w)),
        "within_identity_pct_above_0.7": float((w > 0.7).mean() * 100),
        "between_identity_mean_r": float(b.mean()),
        "between_identity_n_pairs": int(len(b)),
        "between_identity_pct_above_0.7": float((b > 0.7).mean() * 100),
        "identity_signal_present": bool(w.mean() > b.mean()),
    }


def per_identity_fingerprint_stability(
    substrates: Iterable[IdentitySubstrate],
    arm: str = "A",
) -> Dict[str, float]:
    """
    Per-identity cross-model 6-D fingerprint correlation.

    For each identity tested on ≥2 models in `arm`, compute the Pearson r
    of the 6-D fingerprint vector across model pairs. High r (≈ +0.9)
    means the identity's overall fingerprint SHAPE is consistent across
    models, even if individual dimensions wobble. SECI treats this as the
    primary identity-level claim.
    """
    by_id_model = defaultdict(dict)
    for s in substrates:
        if s.metadata.get("arm") != arm:
            continue
        scores = s.dimension_scores
        if all(d in scores for d in DIMENSIONS):
            by_id_model[s.metadata["identity"]][s.metadata["model"]] = scores

    rs_overall: List[float] = []
    per_identity: Dict[str, List[float]] = {}
    for iid, mod_scores in by_id_model.items():
        models = list(mod_scores)
        if len(models) < 2:
            continue
        per_id_rs: List[float] = []
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                v1 = np.array([mod_scores[m1][d] for d in DIMENSIONS])
                v2 = np.array([mod_scores[m2][d] for d in DIMENSIONS])
                if np.std(v1) == 0 or np.std(v2) == 0:
                    continue
                r, _ = pearsonr(v1, v2)
                per_id_rs.append(r)
        if per_id_rs:
            per_identity[iid] = per_id_rs
            rs_overall.extend(per_id_rs)

    if not rs_overall:
        return {}
    arr = np.array(rs_overall)
    return {
        "mean_r": float(arr.mean()),
        "median_r": float(np.median(arr)),
        "n_pairs": int(len(arr)),
        "pct_above_0.7": float((arr > 0.7).mean() * 100),
        "per_identity_mean_r": {
            iid: float(np.mean(rs)) for iid, rs in per_identity.items()
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pair_arms_by_cell(
    substrates: Iterable[IdentitySubstrate],
    arm_x: str, arm_y: str,
) -> List[Tuple[IdentitySubstrate, IdentitySubstrate]]:
    """Match arm_x and arm_y records by (model, identity)."""
    by_cell: Dict[Tuple[str, str], Dict[str, IdentitySubstrate]] = defaultdict(dict)
    for s in substrates:
        cell = (s.metadata.get("model", "?"), s.metadata.get("identity", "?"))
        by_cell[cell][s.metadata.get("arm", "?")] = s
    out: List[Tuple[IdentitySubstrate, IdentitySubstrate]] = []
    for cell, by_arm in by_cell.items():
        if arm_x in by_arm and arm_y in by_arm:
            out.append((by_arm[arm_x], by_arm[arm_y]))
    return out


def _classify_rank_consistency(r: float) -> str:
    if r >= 0.5:
        return "substrate-portable"
    if r >= 0.3:
        return "modest"
    if r >= 0.1:
        return "weak"
    if r >= -0.1:
        return "chance"
    return "inverse"
