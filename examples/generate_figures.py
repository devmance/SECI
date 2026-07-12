#!/usr/bin/env python3
"""
Regenerate the three SECI v2.2 paper figures from the published
validation outputs.

  fig1_three_claims            — Claim A / Claim B paired deltas (bars ± SD)
                                 with Claim C cross-model identity-ranking r
                                 (diamonds, right axis) per dimension.
  fig2_variance_decomposition  — between-identity vs between-model SD per
                                 dimension, with the model/identity ratio
                                 annotated above each group.
  fig3_fingerprint_stability   — per-identity cross-model fingerprint
                                 correlation (arm A), with summary statistics.

All numbers are read from validation_outputs/*.json. For the per-identity
panel of fig3, the script uses the per-identity means embedded in
fingerprint_stability.json; if --data-dir is given (a directory of
per-session *_analysis.json files), the per-identity correlations are
recomputed from raw fingerprint vectors and cross-checked against the
validation output.

Usage:
    python examples/generate_figures.py
    python examples/generate_figures.py --data-dir /path/to/analysis_jsons
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIMENSIONS = ("ICT", "NCG", "PD", "TP", "CCC", "DEA")

# Palette (matches the published figures)
COLOR_CLAIM_A = "#1b4965"   # dark navy
COLOR_CLAIM_B = "#62b6cb"   # light blue
COLOR_CLAIM_C = "#bc4b51"   # red diamonds / right axis
COLOR_IDENTITY_SD = "#2d8659"  # green
COLOR_MODEL_SD = "#d7572a"     # orange
COLOR_RHO_PLAIN = "#666666"
COLOR_RHO_FLAG = "#d7572a"
COLOR_STAT_LABEL = "#555555"

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11,
    "axes.axisbelow": True,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


def _load(validation_dir: Path, name: str) -> dict:
    with open(validation_dir / name) as f:
        return json.load(f)


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.8)


def _save(fig, out_dir: Path, stem: str):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote {out_dir / stem}.png / .pdf")


# ---------------------------------------------------------------------------
# Figure 1 — the three claims
# ---------------------------------------------------------------------------

def fig1_three_claims(claim_a: dict, claim_b: dict, claim_c: dict,
                      out_dir: Path):
    a_mean = [claim_a[d]["mean"] for d in DIMENSIONS]
    a_sd = [claim_a[d]["sd"] for d in DIMENSIONS]
    b_mean = [claim_b[d]["mean"] for d in DIMENSIONS]
    b_sd = [claim_b[d]["sd"] for d in DIMENSIONS]
    c_r = [claim_c[d]["mean_r"] for d in DIMENSIONS]

    x = np.arange(len(DIMENSIONS))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    ax.bar(x - 0.30, a_mean, width, yerr=a_sd, capsize=4,
           error_kw={"linewidth": 1.4}, color=COLOR_CLAIM_A,
           label="Claim A — framework (a−c)")
    ax.bar(x - 0.06, b_mean, width, yerr=b_sd, capsize=4,
           error_kw={"linewidth": 1.4}, color=COLOR_CLAIM_B,
           label="Claim B — base-null (a−b)")
    ax.axhline(0, color="#aaaaaa", linewidth=1.0, linestyle="--", zorder=1)

    ylim = 58  # symmetric so 0 aligns with r = 0 on the right axis
    ax.set_ylim(-ylim, ylim)
    ax.set_yticks(np.arange(-40, 41, 20))
    ax.set_xticks(x)
    ax.set_xticklabels(DIMENSIONS)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Paired delta on a 0–100 dimension scale")
    _style_axes(ax)

    ax2 = ax.twinx()
    ax2.scatter(x + 0.33, c_r, marker="D", s=70, color=COLOR_CLAIM_C,
                zorder=5, label="Claim C — cross-model r")
    ax2.set_ylim(-1, 1)
    ax2.set_yticks(np.arange(-1.0, 1.01, 0.25))
    ax2.set_ylabel("Cross-model identity-ranking r", color=COLOR_CLAIM_C)
    ax2.tick_params(axis="y", colors=COLOR_CLAIM_C)
    ax2.spines["right"].set_color(COLOR_CLAIM_C)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.05),
               frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, "fig1_three_claims")


# ---------------------------------------------------------------------------
# Figure 2 — variance decomposition
# ---------------------------------------------------------------------------

def fig2_variance_decomposition(var_decomp: dict, out_dir: Path):
    id_sd = [var_decomp[d]["between_identity_sd"] for d in DIMENSIONS]
    mod_sd = [var_decomp[d]["between_model_sd"] for d in DIMENSIONS]
    ratio = [var_decomp[d]["model_to_identity_ratio"] for d in DIMENSIONS]

    x = np.arange(len(DIMENSIONS))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width / 2, id_sd, width, color=COLOR_IDENTITY_SD,
           label="between-identity SD")
    ax.bar(x + width / 2, mod_sd, width, color=COLOR_MODEL_SD,
           label="between-model SD")

    ymax = max(max(id_sd), max(mod_sd)) * 1.22
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax, 2))
    for xi, rho in zip(x, ratio):
        flagged = rho > 1.0
        ax.text(xi, ymax * 0.965, f"ρ = {rho:.2f}×",
                ha="center", va="top",
                color=COLOR_RHO_FLAG if flagged else COLOR_RHO_PLAIN,
                fontweight="bold" if flagged else "normal", fontsize=11.5)

    ax.set_xticks(x)
    ax.set_xticklabels(DIMENSIONS)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Standard deviation (on 0–100 scale)")
    _style_axes(ax)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.32),
              frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_variance_decomposition")


# ---------------------------------------------------------------------------
# Figure 3 — fingerprint stability
# ---------------------------------------------------------------------------

def fig3_fingerprint_stability(fps: dict, out_dir: Path, discriminant: dict | None = None):
    per_identity = fps["per_identity_mean_r"]
    order = sorted(per_identity, key=per_identity.get, reverse=True)
    values = [per_identity[i] for i in order]
    mean_r = fps["mean_r"]
    median_r = fps["median_r"]
    n_pairs = fps["n_pairs"]
    pct_above = fps["pct_above_0.7"]

    fig = plt.figure(figsize=(7.9, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.42)

    ax = fig.add_subplot(gs[0])
    y = np.arange(len(order))
    ax.barh(y, values, height=0.72, color=COLOR_CLAIM_A)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_ylim(len(order) + 0.15, -0.55)  # inverted; bottom strip for legend
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Mean Pearson r across model pairs")
    ax.set_title("Per-identity (sorted)", loc="left", fontsize=12.5)
    ax.axvline(0.7, ymin=0.11, color="#b0b0b0", linewidth=1.2,
               linestyle="--", label="r = +0.7")
    ax.axvline(mean_r, ymin=0.11, color=COLOR_CLAIM_C, linewidth=1.6,
               alpha=0.7, label=f"mean r = {mean_r:+.3f}")
    if discriminant:
        between = discriminant["between_identity_mean_r"]
        ax.axvline(between, ymin=0.11, color="#e08b3c", linewidth=1.6,
                   linestyle=":", alpha=0.9,
                   label="between-identity baseline")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=3,
              frameon=False, fontsize=9, handlelength=1.3,
              columnspacing=0.9)

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    stats = [
        ("Mean cross-model r", f"{mean_r:+.3f}"),
        ("Median", f"{median_r:+.3f}"),
        ("Cross-model pairs", f"{n_pairs}"),
        ("Pairs with r > +0.7", f"{pct_above:.0f}%"),
        ("Identities (≥2 models)", f"{len(order)}"),
    ]
    if discriminant:
        stats.append(("Between-identity mean r",
                      f"{discriminant['between_identity_mean_r']:+.3f}"))
    for i, (label, value) in enumerate(stats):
        ytop = 1.0 - i * (1.0 / len(stats))
        ax2.text(0, ytop, label, transform=ax2.transAxes, fontsize=11.5,
                 color=COLOR_STAT_LABEL, va="top")
        ax2.text(0, ytop - 0.065, value, transform=ax2.transAxes,
                 fontsize=19, color=COLOR_CLAIM_A, fontweight="bold",
                 va="top")

    _save(fig, out_dir, "fig3_fingerprint_stability")


# ---------------------------------------------------------------------------
# Optional recomputation of fig3 per-identity stability from raw analysis
# JSONs (cross-checked against fingerprint_stability.json)
# ---------------------------------------------------------------------------

def compute_fingerprint_stability(data_dir: Path, arm: str = "A") -> dict:
    """Recompute per-identity cross-model fingerprint correlations from a
    directory tree of per-session *_analysis.json files."""
    by_id_model: dict = defaultdict(dict)
    for path in sorted(data_dir.rglob("*_analysis.json")):
        with open(path) as f:
            rec = json.load(f)
        meta = rec.get("session_metadata", {})
        if meta.get("arm") != arm:
            continue
        vec = rec.get("fingerprint_vector", {})
        if all(d in vec for d in DIMENSIONS):
            by_id_model[meta["identity_id"]][meta["model"]] = vec

    rs_overall = []
    per_identity = {}
    for iid, mod_scores in by_id_model.items():
        models = list(mod_scores)
        if len(models) < 2:
            continue
        per_id_rs = []
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                v1 = np.array([mod_scores[m1][d] for d in DIMENSIONS])
                v2 = np.array([mod_scores[m2][d] for d in DIMENSIONS])
                if np.std(v1) == 0 or np.std(v2) == 0:
                    continue
                per_id_rs.append(float(np.corrcoef(v1, v2)[0, 1]))
        if per_id_rs:
            per_identity[iid] = per_id_rs
            rs_overall.extend(per_id_rs)

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


def cross_check(computed: dict, published: dict):
    ok = True
    for key in ("mean_r", "median_r", "n_pairs", "pct_above_0.7"):
        if not math.isclose(computed[key], published[key],
                            rel_tol=0, abs_tol=1e-6):
            ok = False
            print(f"  MISMATCH {key}: recomputed {computed[key]} vs "
                  f"published {published[key]}")
    for iid, r in published["per_identity_mean_r"].items():
        rc = computed["per_identity_mean_r"].get(iid)
        if rc is None or not math.isclose(rc, r, rel_tol=0, abs_tol=1e-6):
            ok = False
            print(f"  MISMATCH per-identity {iid}: recomputed {rc} vs "
                  f"published {r}")
    print("  cross-check vs fingerprint_stability.json: "
          + ("OK" if ok else "FAILED"))
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------------------

def main():
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validation-dir", type=Path,
                    default=repo_root / "validation_outputs",
                    help="Directory of validation output JSONs")
    ap.add_argument("--out-dir", type=Path, default=repo_root / "figures",
                    help="Directory to write figures to")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Optional directory of per-session *_analysis.json "
                         "files; recomputes fig3 per-identity stability and "
                         "cross-checks it against the validation output")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    claim_a = _load(args.validation_dir, "claim_a_population.json")
    claim_b = _load(args.validation_dir, "claim_b_population.json")
    claim_c = _load(args.validation_dir, "claim_c_cross_model.json")
    var_decomp = _load(args.validation_dir, "variance_decomposition.json")
    fps = _load(args.validation_dir, "fingerprint_stability.json")

    if args.data_dir is not None:
        print(f"Recomputing fingerprint stability from {args.data_dir}")
        computed = compute_fingerprint_stability(args.data_dir)
        cross_check(computed, fps)
        fps = computed

    print("Figure 1 (three claims):")
    for d in DIMENSIONS:
        print(f"  {d}: A {claim_a[d]['mean']:+.2f} ({claim_a[d]['sd']:.2f})"
              f"  B {claim_b[d]['mean']:+.2f} ({claim_b[d]['sd']:.2f})"
              f"  C r {claim_c[d]['mean_r']:+.3f}")
    fig1_three_claims(claim_a, claim_b, claim_c, args.out_dir)

    print("Figure 2 (variance decomposition):")
    for d in DIMENSIONS:
        v = var_decomp[d]
        print(f"  {d}: identity SD {v['between_identity_sd']:.2f}  "
              f"model SD {v['between_model_sd']:.2f}  "
              f"ρ {v['model_to_identity_ratio']:.2f}×")
    fig2_variance_decomposition(var_decomp, args.out_dir)

    print("Figure 3 (fingerprint stability):")
    print(f"  mean r {fps['mean_r']:+.3f}  median {fps['median_r']:+.3f}  "
          f"pairs {fps['n_pairs']}  >0.7 {fps['pct_above_0.7']:.0f}%  "
          f"identities {len(fps['per_identity_mean_r'])}")
    try:
        discriminant = _load(args.validation_dir, "fingerprint_discriminant.json")
    except FileNotFoundError:
        discriminant = None
    fig3_fingerprint_stability(fps, args.out_dir, discriminant)


if __name__ == "__main__":
    main()
