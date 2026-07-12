"""
Apply SECI's three-claim and variance-decomposition analysis to a
multi-model multi-arm benchmark dataset.

Walks all SECI analysis JSONs at the configured path, loads them as
LLMSubstrate instances, and produces:

  - claim_a_population.json     — arm_a vs arm_c, framework contribution
  - claim_b_population.json     — arm_a vs arm_b, base-model null
  - claim_c_cross_model.json    — per-dimension cross-model identity rankings
  - variance_decomposition.json — between-identity vs between-model SD
  - fingerprint_stability.json  — per-identity 6-D fingerprint r across models
  - warning_flags.json          — auto-generated diagnostic warnings

Usage:
    python -m examples.rescore_dataset \\
        --data-dir /path/to/analysis \\
        --output-dir validation_outputs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

# Allow running from repo root without installing the package
THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1] / "src"))

from seci.substrate import LLMSubstrate
from seci.analysis import (
    population_claim_a,
    population_claim_a_per_model,
    population_claim_b,
    claim_c_cross_model_ranking,
    per_identity_fingerprint_stability,
    fingerprint_discriminant_control,
    variance_decomposition,
    warning_flags,
)


def load_substrates(data_dir: Path) -> List[LLMSubstrate]:
    substrates: List[LLMSubstrate] = []
    seen = set()
    for root, _dirs, files in os.walk(data_dir):
        for name in files:
            if not name.endswith(".json"):
                continue
            if name in seen:
                continue
            path = Path(root) / name
            try:
                sub = LLMSubstrate(path)
            except Exception as e:
                print(f"  skip {path.name}: {e}")
                continue
            if not sub.dimension_scores:
                continue
            seen.add(name)
            substrates.append(sub)
    return substrates


def _tally_verified_terms(data_dir: Path) -> dict:
    """Count consensus-verified novel terms per arm across the dataset."""
    out: dict = {}
    for root, _dirs, files in os.walk(data_dir):
        for name in files:
            if not name.endswith(".json"):
                continue
            try:
                doc = json.loads((Path(root) / name).read_text())
            except Exception:
                continue
            md = doc.get("session_metadata", {})
            arm = md.get("arm") or ("B" if "arm_b" in root else "?")
            terms = (
                doc.get("dimension_details", {}).get("NCG", {}).get("verified_novel_terms")
            )
            if terms is None:
                continue
            bucket = out.setdefault(arm, {"sessions": 0, "verified_terms": 0, "terms": []})
            bucket["sessions"] += 1
            bucket["verified_terms"] += len(terms)
            bucket["terms"].extend(terms)
    for arm, b in out.items():
        b["terms_per_session"] = round(b["verified_terms"] / b["sessions"], 3) if b["sessions"] else 0.0
    return out


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"  wrote {path.relative_to(path.parents[1])}")


def main():
    parser = argparse.ArgumentParser(description="Apply SECI analysis to a benchmark dataset.")
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Path to SECI analysis output directory (contains arm_a/, arm_b/, arm_c/, ...)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("validation_outputs"),
        help="Where to write outputs (default: validation_outputs/)",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: data dir does not exist: {args.data_dir}")
        sys.exit(1)

    print(f"Loading substrates from {args.data_dir}")
    substrates = load_substrates(args.data_dir)
    print(f"Loaded {len(substrates)} substrates")
    arms = sorted({s.metadata.get("arm", "?") for s in substrates})
    models = sorted({s.metadata.get("model", "?") for s in substrates})
    identities = sorted({s.metadata.get("identity", "?") for s in substrates})
    print(f"  arms:       {arms}")
    print(f"  models:     {len(models)}")
    print(f"  identities: {len(identities)}")

    out = args.output_dir
    print(f"\nWriting outputs to {out}")

    # ----- Claim A: framework contribution (arm_a vs arm_c) ------------
    claim_a = population_claim_a(substrates)
    write_json(out / "claim_a_population.json", claim_a)

    # ----- Claim A per model (backs any "positive on all N models" claim)
    claim_a_per_model = population_claim_a_per_model(substrates)
    write_json(out / "claim_a_per_model.json", claim_a_per_model)

    # ----- Claim B: base-model null (arm_a vs arm_b) -------------------
    claim_b = population_claim_b(substrates, scaffolded_arm="A")
    write_json(out / "claim_b_population.json", claim_b)

    # ----- Claim C: cross-model identity ranking -----------------------
    claim_c = claim_c_cross_model_ranking(substrates, arm="A")
    write_json(out / "claim_c_cross_model.json", claim_c)

    # ----- Variance decomposition --------------------------------------
    decomp = variance_decomposition(substrates, arm="A")
    write_json(out / "variance_decomposition.json", decomp)

    # ----- Per-identity fingerprint stability --------------------------
    stability = per_identity_fingerprint_stability(substrates, arm="A")
    write_json(out / "fingerprint_stability.json", stability)

    # ----- Fingerprint discriminant control -----------------------------
    discriminant = fingerprint_discriminant_control(substrates, arm="A")
    write_json(out / "fingerprint_discriminant.json", discriminant)

    # ----- Verified novel term counts per arm ---------------------------
    verified = _tally_verified_terms(args.data_dir)
    write_json(out / "verified_terms.json", verified)

    # ----- Auto-generated warnings -------------------------------------
    flags = warning_flags(decomp, claim_c)
    write_json(out / "warning_flags.json", {"warnings": flags})

    # ----- Console summary ---------------------------------------------
    print("\n" + "=" * 100)
    print("SECI ANALYSIS SUMMARY  (arm_a unless noted)")
    print("=" * 100)
    print(f"\n{'Dim':5s}  {'Claim A (a−c)':>16s}  {'Claim B (a−b)':>16s}  {'Claim C (cross-model r)':>26s}  {'Variance verdict':>32s}")
    print("-" * 100)
    for d in ("ICT", "NCG", "PD", "TP", "CCC", "DEA"):
        a = claim_a.get(d, {})
        b = claim_b.get(d, {})
        c = claim_c.get(d, {})
        v = decomp.get(d, {})
        a_str = f"{a.get('mean', 0):+6.2f} ± {a.get('sd', 0):4.2f}" if a else "—"
        b_str = f"{b.get('mean', 0):+6.2f} ± {b.get('sd', 0):4.2f}" if b else "—"
        c_str = f"r = {c.get('mean_r', 0):+.3f} ({c.get('verdict', '?')})" if c else "—"
        v_str = v.get("verdict", "—")
        print(f"  {d:3s}  {a_str:>16s}  {b_str:>16s}  {c_str:>26s}  {v_str:>32s}")

    print(f"\nPer-identity 6-D fingerprint stability across models:")
    if stability:
        print(f"  mean r = {stability['mean_r']:+.3f}  (median {stability['median_r']:+.3f})")
        print(f"  n cross-model pairs = {stability['n_pairs']}")
        print(f"  % pairs with r > +0.7 = {stability['pct_above_0.7']:.0f}%")

    print(f"\nDiagnostic warnings ({len(flags)}):")
    for f in flags:
        print(f"  • {f}")


if __name__ == "__main__":
    main()
