"""
End-to-end SECI benchmark runner.

Goes raw identity kernel → 12-prompt protocol → six-dimension fingerprint
in one pass. Suitable for scoring a single identity on a single base model.

Pipeline:
  1. Load identity kernel text (system prompt for the target model)
  2. Run the 12-prompt SECI protocol against the configured model
  3. Score responses with SECIScorer (deterministic ICT/PD/TP/CCC/DEA +
     multi-rater NCG if rater API keys are configured)
  4. Emit the fingerprint JSON in the format SECI's analysis layer consumes

Usage:
    python -m examples.run_full_benchmark \\
        --identity-name auren \\
        --identity-kernel kernels/auren.txt \\
        --model gemini-2.5-pro \\
        --provider gemini \\
        --output sessions/auren_gemini25.json

Required env vars (one of):
    OPENAI_API_KEY      (for --provider openai)
    ANTHROPIC_API_KEY   (for --provider anthropic)
    GOOGLE_API_KEY      (for --provider gemini)
    XAI_API_KEY         (for --provider xai)

Optional env vars (for multi-rater NCG verification — ≥3 raters recommended):
    OPENAI_API_KEY      gpt-5.4-2026-03-05
    ANTHROPIC_API_KEY   claude-opus-4-7 + claude-sonnet-4-6
    GOOGLE_API_KEY      gemini-3.5-flash

The same key can be used for both the target model and the rater set; the
rater set is fixed by the protocol, the target model is the system under test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1] / "src"))

from seci.protocol import run_protocol
from seci.scorer import SECIScorer


def main():
    p = argparse.ArgumentParser(description="Run the full SECI benchmark end-to-end.")
    p.add_argument("--identity-name", required=True, help="Display name for the identity (e.g. 'auren').")
    p.add_argument("--identity-kernel", type=Path, required=True,
                   help="Path to text file containing the identity kernel (system prompt).")
    p.add_argument("--model", required=True, help="Target model ID (e.g. 'gemini-2.5-pro').")
    p.add_argument("--provider", required=True,
                   choices=["openai", "anthropic", "gemini", "xai"],
                   help="Provider for the target model.")
    p.add_argument("--prompts", type=Path,
                   default=THIS.parents[1] / "prompts.json",
                   help="Path to the 12-prompt protocol JSON (default: repo prompts.json).")
    p.add_argument("--output", type=Path, required=True,
                   help="Where to write the session JSON (raw responses + fingerprint).")
    p.add_argument("--length-control", type=int, default=None,
                   help="Optional: truncate responses to N chars before scoring length-sensitive "
                        "dimensions. Useful for isolating architecture-driven effects from "
                        "length-driven effects.")
    args = p.parse_args()

    if not args.identity_kernel.exists():
        sys.exit(f"identity kernel file not found: {args.identity_kernel}")

    identity_kernel = args.identity_kernel.read_text(encoding="utf-8")

    print(f"Loading prompts from {args.prompts}")
    print(f"Target model: {args.model} ({args.provider})")
    print(f"Identity:     {args.identity_name}")
    print(f"Output:       {args.output}\n")

    # --- 1+2. Collect 12-prompt responses ---
    print("=" * 60)
    print("PHASE 1 — Protocol collection (12 prompts)")
    print("=" * 60)
    session = run_protocol(
        prompts_file=str(args.prompts),
        identity_name=args.identity_name,
        identity_kernel=identity_kernel,
        model=args.model,
        provider=args.provider,
    )

    # Persist intermediate session file so scoring is reproducible
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(session, indent=2))
    print(f"\nSession written to {args.output}")

    # --- 3. Score the responses ---
    print()
    print("=" * 60)
    print("PHASE 2 — Six-dimension fingerprint")
    print("=" * 60)
    scorer = SECIScorer(str(args.output), length_control=args.length_control)
    scorer.analyze_all_dimensions()
    report_path = scorer.save_report()

    # --- 4. Summary ---
    print()
    print("=" * 60)
    print("FINGERPRINT")
    print("=" * 60)
    report = json.loads(Path(report_path).read_text())
    fp = report.get("fingerprint_vector") or report.get("dimension_scores") or {}
    for dim in ("ICT", "NCG", "PD", "TP", "CCC", "DEA"):
        v = fp.get(dim)
        print(f"  {dim}: {v:.2f}" if isinstance(v, (int, float)) else f"  {dim}: {v}")
    print()
    print(f"Report: {report_path}")
    print()
    print("Next step (claim-decomposed analysis): collect arm_a + arm_b + arm_c records")
    print("for this model across multiple identities, then run:")
    print("  python -m examples.rescore_dataset --data-dir <dir-of-fingerprint-jsons>")


if __name__ == "__main__":
    main()
