"""
Re-run multi-rater NCG verification over an existing SECI dataset.

For every scored analysis JSON, this driver re-executes only the
rater-dependent NCG pipeline (candidate extraction → stage-1 type
classification → stage-2 novelty verification) against the raw session
transcript, then reassembles the NCG score from the freshly verified
novelty count and the stored deterministic subcomponents
(semantic_novelty, framework_construction, concept_emergence). The five
deterministic dimensions are untouched, so no embedding drift is
introduced.

Use this when the rater configuration was degraded in a prior run (e.g.
a rater returning null on every call) and the NCG votes need to be
collected again without recollecting sessions.

Usage:
    python -m examples.rerate_ncg \\
        --raw-dir /path/to/data \\
        --analysis-dir /path/to/data/analysis \\
        --output-dir /path/to/rerated_analysis \\
        [--sessions N] [--max-terms N] [--workers N]

Raw session files are matched to analysis files by name: an analysis file
<name>_analysis.json in <analysis-dir>/<sub>/ must have its raw session at
<raw-dir>/<sub>/<name>.json.

Rater API keys are read from the environment (see seci.scorer.analyzer
RATER_SET): OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1] / "src"))

from seci.scorer.analyzer import (
    analyze_ncg_v22,
    consensus_threshold,
    get_active_raters,
    verified_novelty_score,
)

NCG_WEIGHTS = {"verified_novelty": 0.40, "semantic_novelty": 0.20,
               "framework_construction": 0.20, "concept_emergence": 0.20}


def find_pairs(raw_dir: Path, analysis_dir: Path) -> List[Tuple[Path, Path]]:
    """Match every *_analysis.json to its raw session file."""
    pairs: List[Tuple[Path, Path]] = []
    missing: List[Path] = []
    for analysis_path in sorted(analysis_dir.rglob("*_analysis.json")):
        rel = analysis_path.relative_to(analysis_dir)
        raw_name = analysis_path.name.replace("_analysis.json", ".json")
        raw_path = raw_dir / rel.parent / raw_name
        if raw_path.exists():
            pairs.append((raw_path, analysis_path))
        else:
            missing.append(analysis_path)
    if missing:
        print(f"[!] {len(missing)} analysis files have no matching raw session (skipped):")
        for m in missing[:10]:
            print(f"    {m}")
    return pairs


def ncg_responses_from_raw(raw_path: Path) -> List[Dict]:
    session = json.loads(raw_path.read_text())
    out: List[Dict] = []
    for c in session.get("conversations", []):
        if str(c.get("dimension", "")).startswith("NCG"):
            resp = c.get("responses", {}).get("identity", {})
            if resp.get("success") and resp.get("content"):
                out.append(resp)
    return out


def rerate_one(
    raw_path: Path,
    analysis_path: Path,
    out_path: Path,
    raters: List[Dict[str, str]],
    max_terms: Optional[int],
) -> Dict[str, Any]:
    analysis = json.loads(analysis_path.read_text())
    old_ncg = analysis.get("dimension_details", {}).get("NCG", {})
    old_final = analysis.get("fingerprint_vector", {}).get("NCG")

    result = analyze_ncg_v22(ncg_responses_from_raw(raw_path), raters, max_terms=max_terms)

    vn_score = verified_novelty_score(len(result["verified_novel_terms"]))
    deterministic = {
        k: float(old_ncg.get(k, 0.0))
        for k in ("semantic_novelty", "framework_construction", "concept_emergence")
    }
    new_final = round(
        vn_score * NCG_WEIGHTS["verified_novelty"]
        + deterministic["semantic_novelty"] * NCG_WEIGHTS["semantic_novelty"]
        + deterministic["framework_construction"] * NCG_WEIGHTS["framework_construction"]
        + deterministic["concept_emergence"] * NCG_WEIGHTS["concept_emergence"],
        2,
    )

    analysis["fingerprint_vector"]["NCG"] = new_final
    analysis["dimension_details"]["NCG"] = {
        "verified_novelty": round(vn_score, 2),
        **{k: round(v, 2) for k, v in deterministic.items()},
        "final_score": new_final,
        "verified_novel_terms": result["verified_novel_terms"],
        "candidates_extracted": result["candidates_extracted"],
        "stage_1_results": result["stage_1_results"],
        "stage_2_results": result["stage_2_results"],
        "rater_agreement": result["rater_agreement"],
        "verifier_set": result["verifier_set"],
        "consensus_threshold_used": result.get("consensus_threshold_used"),
        "ncg_rerate": {
            "rerated_at": datetime.now().isoformat(),
            "previous_final_score": old_final,
            "previous_verified_novelty": old_ncg.get("verified_novelty"),
        },
    }
    analysis["rater_set_active"] = [r["id"] for r in raters]
    # Provenance pointer to a document that was never published; drop it.
    analysis.pop("pre_registration", None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(analysis, indent=2, default=str))

    null_counts: Dict[str, int] = {r["id"]: 0 for r in raters}
    rounds = 0
    for s1 in result["stage_1_results"]:
        rounds += 1
        for rid, v in s1["stage_1_classifications"].items():
            if v is None:
                null_counts[rid] = null_counts.get(rid, 0) + 1
    return {
        "file": analysis_path.name,
        "old_ncg": old_final,
        "new_ncg": new_final,
        "verified_terms": result["verified_novel_terms"],
        "stage_1_rounds": rounds,
        "stage_1_nulls": null_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Re-run multi-rater NCG over a SECI dataset.")
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="Directory containing raw session arm_*/ subdirectories.")
    parser.add_argument("--analysis-dir", type=Path, required=True,
                        help="Directory containing scored *_analysis.json files (arm subdirs).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write re-rated analysis JSONs (mirrors analysis-dir layout).")
    parser.add_argument("--sessions", type=int, default=None,
                        help="Only process the first N sessions (smoke tests).")
    parser.add_argument("--max-terms", type=int, default=None,
                        help="Cap candidate terms per session (smoke tests only).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Sessions processed concurrently (default 4).")
    args = parser.parse_args()

    raters = get_active_raters()
    if len(raters) < 3:
        print(f"ERROR: only {len(raters)} rater(s) configured; the SECI consensus rule "
              f"needs >=3 active raters to ever verify a term. Set more API keys.")
        sys.exit(1)
    print(f"Active raters ({len(raters)}): {[r['id'] for r in raters]}")
    print(f"Consensus threshold: >={consensus_threshold(len(raters))} of {len(raters)}")

    pairs = find_pairs(args.raw_dir, args.analysis_dir)
    if args.sessions is not None:
        pairs = pairs[: args.sessions]
    print(f"Sessions to re-rate: {len(pairs)}")

    summaries: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                rerate_one, raw, analysis,
                args.output_dir / analysis.relative_to(args.analysis_dir),
                raters, args.max_terms,
            ): analysis.name
            for raw, analysis in pairs
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                s = fut.result()
            except Exception as e:
                print(f"[!] {name} FAILED: {str(e)[:200]}")
                continue
            summaries.append(s)
            terms = ", ".join(s["verified_terms"]) or "-"
            print(f"[{len(summaries)}/{len(pairs)}] {s['file']}: NCG {s['old_ncg']} -> {s['new_ncg']}  verified: {terms}")

    # Aggregate rater health across the run
    total_rounds = sum(s["stage_1_rounds"] for s in summaries)
    print("\n" + "=" * 72)
    print(f"Re-rated {len(summaries)}/{len(pairs)} sessions, {total_rounds} stage-1 rounds")
    if total_rounds:
        agg: Dict[str, int] = {}
        for s in summaries:
            for rid, n in s["stage_1_nulls"].items():
                agg[rid] = agg.get(rid, 0) + n
        print("Stage-1 null votes per rater:")
        for rid, n in sorted(agg.items()):
            print(f"  {rid:26s} {n:5d} / {total_rounds}  ({100.0 * n / total_rounds:.1f}%)")
    moved = [s for s in summaries if s["old_ncg"] != s["new_ncg"]]
    print(f"Sessions with changed NCG: {len(moved)}/{len(summaries)}")
    report_path = args.output_dir / "rerate_summary.json"
    report_path.write_text(json.dumps(summaries, indent=2))
    print(f"Summary written to {report_path}")


if __name__ == "__main__":
    main()
