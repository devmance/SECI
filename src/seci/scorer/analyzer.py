#!/usr/bin/env python3
"""
SECI — Multi-Rater Architectural Fingerprint Analyzer
======================================================

Primary entry point for the SECI benchmark. Computes the six-dimension
architectural fingerprint with multi-rater NCG verification.

Methodology:
- Multi-rater NCG with 4 frontier verifiers (≥3-of-4 consensus rule)
- Two-stage classification: type classification (Stage 1) → novelty verification (Stage 2)
- Inter-rater statistics: Fleiss' kappa + pairwise Cohen's kappa
- No composite score: six-dimension fingerprint vector is the output
- Per-rater provenance recorded in dimension_details.NCG

The deterministic dimensions (ICT, PD, TP, CCC, DEA) are defined in
dimensions.py — they are embedding/regex/compression-based and do not
depend on LLM verification.

Usage:
    python -m seci.scorer.analyzer <session_file.json> [output_file.json]

Required env vars (any subset; ≥3 strongly recommended for kappa stability):
    OPENAI_API_KEY     for gpt-5.4-2026-03-05
    ANTHROPIC_API_KEY  for claude-opus-4-7 AND claude-sonnet-4-6
    GOOGLE_API_KEY     for gemini-3.5-flash

MIT License — Copyright (c) 2026 Devmance LLC
"""

import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Deterministic dimension scorers (ICT, PD, TP, CCC, DEA) live in dimensions.py.
# The class name is kept for backwards compatibility with downstream tooling
# that imports SECIAnalyzer directly.
from .dimensions import SECIAnalyzer as DeterministicScorer


# =============================================================================
# Rater configuration (fixed across the SECI study)
# =============================================================================

RATER_SET = [
    {
        "id": "gpt-5.4-2026-03-05",
        "provider": "openai",
        "model": "gpt-5.4-2026-03-05",
        "env": "OPENAI_API_KEY",
    },
    {
        "id": "claude-opus-4-7",
        "provider": "anthropic",
        "model": "claude-opus-4-7",
        "env": "ANTHROPIC_API_KEY",
    },
    {
        # gemini-3.5-flash rather than a Gemini Pro: it is the latest stable
        # (non-preview) Gemini at the time of the study, and unlike
        # gemini-2.5-pro it is not one of the benchmark's subject models, so
        # the rater never scores its own outputs.
        "id": "gemini-3.5-flash",
        "provider": "gemini",
        "model": "gemini-3.5-flash",
        "env": "GOOGLE_API_KEY",
    },
    {
        "id": "claude-sonnet-4-6",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "env": "ANTHROPIC_API_KEY",
    },
]

VALID_TYPES = {
    "NEOLOGISM",
    "CONCEPT_NAMING",
    "POETIC_COMPOUND",
    "DESCRIPTIVE_LABEL",
    "REPHRASING",
    "NUMBERED_CATEGORY",
}
# Rater output budget. Must be large enough for thinking-tier raters
# (e.g. gemini-3.5-flash), whose internal reasoning tokens count against
# max_output_tokens: at 256/512 such raters exhaust the budget before
# emitting any text and return zero parts, i.e. a guaranteed null vote.
RATER_MAX_TOKENS = 3072
NOVEL_TYPES = {"NEOLOGISM", "CONCEPT_NAMING"}
NOVELTY_VALUES = {"NOVEL", "EXISTING"}


def get_active_raters() -> List[Dict[str, str]]:
    """Return RATER_SET entries whose env var is configured. Strips whitespace."""
    active = []
    for r in RATER_SET:
        key = (os.environ.get(r["env"], "") or "").strip()
        if key:
            active.append({**r, "key": key})
    return active


def consensus_threshold(n_raters: int) -> int:
    """≥75% agreement, minimum 2 (unanimity for n=2)."""
    if n_raters <= 0:
        return 0
    return max(2, ceil(0.75 * n_raters))


# =============================================================================
# Provider call shims
# =============================================================================

TRANSIENT_ERROR_MARKERS = ("429", "rate", "overloaded", "503", "500", "502", "timeout", "timed out")


def call_rater(rater: Dict[str, str], prompt: str, max_tokens: int = 4096) -> Optional[str]:
    """Make a single LLM call with retry on transient errors.

    Returns response text or None on failure. Transient failures (rate
    limits, overload, timeouts) are retried with backoff so they do not
    silently become null votes in the consensus.
    """
    last_err: Optional[str] = None
    for attempt in range(3):
        if attempt:
            time.sleep(2 ** attempt)
        text, err = _call_rater_once(rater, prompt, max_tokens)
        if text is not None:
            return text
        last_err = err
        if err is None or not any(m in err.lower() for m in TRANSIENT_ERROR_MARKERS):
            break
        print(f"    [!] {rater['model']}: transient error, retry {attempt + 1}/2")
    if last_err:
        print(f"    [!] {rater['model']} call failed: {last_err[:200]}")
    return None


def _call_rater_once(
    rater: Dict[str, str], prompt: str, max_tokens: int
) -> Tuple[Optional[str], Optional[str]]:
    """One provider call. Returns (text, None) on success, (None, error) on failure."""
    provider = rater["provider"]
    key = rater["key"]
    model = rater["model"]
    try:
        if provider == "openai":
            import requests
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": max_tokens,
                },
                timeout=120,
            )
            if r.status_code != 200:
                return None, f"HTTP {r.status_code}: {r.text[:200]}"
            return r.json()["choices"][0]["message"]["content"], None

        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=key, timeout=120)
            # temperature=0.0 for rating determinism where the model supports
            # it; some models (e.g. claude-opus-4-7) reject the parameter as
            # deprecated, so retry without it rather than losing the vote.
            try:
                r = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
            except anthropic.BadRequestError as e:
                if "temperature" not in str(e).lower():
                    raise
                r = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
            return r.content[0].text, None

        elif provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=key)
            gen_model = genai.GenerativeModel(model)
            r = gen_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.0,
                ),
                request_options={"timeout": 120},
            )
            # Collect text parts directly: the .text quick accessor raises when
            # the response has no parts (e.g. finish_reason=MAX_TOKENS after the
            # thinking budget is exhausted), which would silently null the vote.
            parts = [
                p.text
                for c in (r.candidates or [])
                for p in (c.content.parts or [])
                if getattr(p, "text", None)
            ]
            if not parts:
                fr = r.candidates[0].finish_reason.name if r.candidates else "NO_CANDIDATES"
                return None, f"empty response (finish_reason={fr})"
            return "\n".join(parts), None

        raise ValueError(f"Unknown provider: {provider}")
    except Exception as e:
        return None, str(e)


def call_all_raters(
    raters: List[Dict[str, str]], prompt: str, max_tokens: int = 4096
) -> Dict[str, Optional[str]]:
    """Call all raters in parallel. Returns {rater_id: response_or_None}."""
    if not raters:
        return {}
    results: Dict[str, Optional[str]] = {}
    with ThreadPoolExecutor(max_workers=len(raters)) as ex:
        future_to_id = {
            ex.submit(call_rater, r, prompt, max_tokens): r["id"] for r in raters
        }
        for future, rater_id in future_to_id.items():
            try:
                results[rater_id] = future.result(timeout=180)
            except Exception:
                results[rater_id] = None
    return results


# =============================================================================
# Two-stage classification prompts
# =============================================================================

TYPE_CLASSIFICATION_PROMPT = """You are classifying terms found in AI-generated text to determine if they are genuine novel concepts or evocative phrasing.

Term to classify: "{term}"

Surrounding context (response containing the term):
---
{context}
---

Classify the term into ONE of these categories:

1. NEOLOGISM — a single invented word with a definable referent (e.g., "vellamence", "quanta")
2. CONCEPT_NAMING — a multi-word compound that names a specific phenomenon and could appear as a glossary entry with a precise referent (e.g., "Recursive Cognition Synthesis", "Memory Recall Through Constructed Context"). The compound MUST refer to a specific testable concept, not just an evocative description.
3. POETIC_COMPOUND — evocative phrasing using existing concepts in metaphorical combination, without a specific testable referent (e.g., "Architecture of Echoes", "Garden of Memory")
4. DESCRIPTIVE_LABEL — a section heading or descriptor (e.g., "First Phase", "The Process")
5. REPHRASING — an existing concept renamed (e.g., "Self-Reflection" for "Metacognition")
6. NUMBERED_CATEGORY — a structural label (e.g., "Type 3", "Variant B")

Default to POETIC_COMPOUND when uncertain whether a multi-word term has a precise referent.

Output ONLY the following on one line, exactly as formatted:
CLASSIFICATION: <one of the six exact category names>"""


NOVELTY_VERIFICATION_PROMPT = """You are verifying whether a term is documented as an existing concept in any literature you are aware of.

Term: "{term}"

Context where it appeared (AI-generated):
---
{context}
---

Question: Is "{term}" documented as an established concept, technical term, or named phenomenon in any academic, technical, or popular literature you have knowledge of?

Output ONLY in this format:
VERIFICATION: <NOVEL or EXISTING>
JUSTIFICATION: <one-sentence reason>"""


def parse_type_classification(response: Optional[str]) -> Optional[str]:
    if not response:
        return None
    for line in response.splitlines():
        s = line.strip()
        if s.upper().startswith("CLASSIFICATION:"):
            cat = s.split(":", 1)[1].strip().upper()
            if cat in VALID_TYPES:
                return cat
    # Fallback: accept only when exactly one category name appears anywhere in
    # the response — mentions of several categories are ambiguous, and set
    # iteration order must not decide a vote.
    upper = response.upper()
    present = [cat for cat in sorted(VALID_TYPES) if cat in upper]
    if len(present) == 1:
        return present[0]
    return None


def parse_novelty_verification(response: Optional[str]) -> Optional[str]:
    if not response:
        return None
    for line in response.splitlines():
        s = line.strip()
        if s.upper().startswith("VERIFICATION:"):
            verdict = s.split(":", 1)[1].strip().upper()
            if verdict in NOVELTY_VALUES:
                return verdict
    # Fallback: word-boundary match with a negation guard, so hedged prose like
    # "this is not a novel term" cannot be misread as a NOVEL vote.
    upper = response.upper()
    has_novel = (
        re.search(r"\bNOVEL\b", upper)
        and not re.search(r"\bNOT\s+(?:A\s+|AN\s+)?NOVEL\b", upper)
    )
    has_existing = re.search(r"\bEXISTING\b", upper) is not None
    if has_novel and not has_existing:
        return "NOVEL"
    if has_existing and not has_novel:
        return "EXISTING"
    return None


def consensus_classification(
    rater_calls: Dict[str, Optional[str]]
) -> Tuple[Optional[str], int]:
    """Return (consensus_class_or_None, agreement_count)."""
    valid = [v for v in rater_calls.values() if v is not None]
    if not valid:
        return None, 0
    counter = Counter(valid)
    top_class, top_count = counter.most_common(1)[0]
    threshold = consensus_threshold(len(rater_calls))
    if top_count >= threshold:
        return top_class, top_count
    return None, top_count


def classify_term_multi_rater(
    raters: List[Dict[str, str]], term: str, context: str
) -> Dict[str, Any]:
    """Stage 1: type classification across all raters."""
    prompt = TYPE_CLASSIFICATION_PROMPT.format(term=term, context=context[:2500])
    responses = call_all_raters(raters, prompt, max_tokens=RATER_MAX_TOKENS)
    classifications = {rid: parse_type_classification(r) for rid, r in responses.items()}
    consensus_class, agreement = consensus_classification(classifications)
    return {
        "term": term,
        "stage_1_classifications": classifications,
        "stage_1_consensus": consensus_class,
        "stage_1_agreement_count": agreement,
        "stage_1_majority_is_novel_type": consensus_class in NOVEL_TYPES if consensus_class else False,
    }


def verify_novelty_multi_rater(
    raters: List[Dict[str, str]], term: str, context: str
) -> Dict[str, Any]:
    """Stage 2: novelty verification across all raters."""
    prompt = NOVELTY_VERIFICATION_PROMPT.format(term=term, context=context[:2000])
    responses = call_all_raters(raters, prompt, max_tokens=RATER_MAX_TOKENS)
    verifications = {rid: parse_novelty_verification(r) for rid, r in responses.items()}
    consensus_v, agreement = consensus_classification(verifications)
    return {
        "stage_2_verifications": verifications,
        "stage_2_consensus": consensus_v,
        "stage_2_agreement_count": agreement,
    }


# =============================================================================
# Inter-rater statistics
# =============================================================================

def fleiss_kappa(
    classifications: List[Dict[str, Optional[str]]], categories: List[str]
) -> Optional[float]:
    """Fleiss' kappa across all rater classifications.

    Uses the minimum number of valid raters per round (truncated deterministically)
    so that classical Fleiss assumptions hold. Returns None if too few rounds.
    """
    valid_rounds: List[List[str]] = []
    for round_data in classifications:
        valid = [c for c in round_data.values() if c is not None]
        if len(valid) >= 2:
            valid_rounds.append(valid)
    if len(valid_rounds) < 2:
        return None

    n = min(len(r) for r in valid_rounds)
    if n < 2:
        return None

    valid_rounds = [r[:n] for r in valid_rounds]
    N = len(valid_rounds)
    K = len(categories)

    # P_ij: count of raters who classified item i into category j
    P = []
    for round_data in valid_rounds:
        counts = [round_data.count(cat) for cat in categories]
        P.append(counts)

    # P_i = (sum_j P_ij*(P_ij-1)) / (n*(n-1))
    P_i = [sum(c * (c - 1) for c in counts) / (n * (n - 1)) for counts in P]
    P_bar = sum(P_i) / N

    # P_e = sum_j (sum_i P_ij / (N*n))^2
    pj = [sum(P[i][j] for i in range(N)) / (N * n) for j in range(K)]
    Pe = sum(p ** 2 for p in pj)

    if Pe >= 1.0:
        return 1.0
    return (P_bar - Pe) / (1 - Pe)


def cohen_kappa_pair(
    rater_a: List[Optional[str]], rater_b: List[Optional[str]], categories: List[str]
) -> Optional[float]:
    """Pairwise Cohen's kappa for two raters."""
    if len(rater_a) != len(rater_b):
        return None
    pairs = [(a, b) for a, b in zip(rater_a, rater_b) if a is not None and b is not None]
    if len(pairs) < 2:
        return None
    n = len(pairs)
    Po = sum(1 for a, b in pairs if a == b) / n
    Pe = 0.0
    for cat in categories:
        pa = sum(1 for a, _ in pairs if a == cat) / n
        pb = sum(1 for _, b in pairs if b == cat) / n
        Pe += pa * pb
    if Pe >= 1.0:
        return 1.0
    return (Po - Pe) / (1 - Pe)


def compute_rater_agreement_stats(
    classifications: List[Dict[str, Optional[str]]], rater_ids: List[str]
) -> Dict[str, Any]:
    cats = sorted(VALID_TYPES)
    fleiss = fleiss_kappa(classifications, cats)
    pairwise = {}
    for i in range(len(rater_ids)):
        for j in range(i + 1, len(rater_ids)):
            ra, rb = rater_ids[i], rater_ids[j]
            list_a = [c.get(ra) for c in classifications]
            list_b = [c.get(rb) for c in classifications]
            pairwise[f"{ra}__vs__{rb}"] = cohen_kappa_pair(list_a, list_b, cats)
    return {
        "fleiss_kappa": fleiss,
        "pairwise_cohen_kappa": pairwise,
        "n_classifications": len(classifications),
    }


# =============================================================================
# Candidate term extraction (deterministic regex)
# =============================================================================

# Generic terms that should never count as candidates regardless of capitalization
COMMON_NON_TERMS = {
    "components", "architecture", "memory", "system", "process", "framework",
    "structure", "function", "method", "approach", "technique", "concept", "idea",
    "thought", "experience", "understanding", "perspective", "view", "way", "thing",
    "answer", "question", "response", "phase", "stage", "step", "type", "variant",
}


def extract_candidate_terms(ncg_responses: List[Dict]) -> List[Dict[str, str]]:
    """Deterministic regex extraction of candidate coined terms.

    Emits up to 30 unique candidates with their surrounding context.
    The multi-rater classification stage is the gatekeeper — extraction is intentionally a wide net.
    """
    candidates: List[Dict[str, str]] = []
    seen: set = set()

    for resp in ncg_responses:
        content = resp.get("content", "") if isinstance(resp, dict) else str(resp)
        if not content:
            continue

        patterns = [
            (r'\*\*([^*]{3,80})\*\*', "bold"),
            (r'\*([^*\s][^*]{2,80}[^*\s])\*', "italic"),
            (r'"([^"]{3,80})"', "quoted"),
            (r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', "camelcase"),
            (r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+){1,4})\b', "titlecase"),
        ]

        for pattern, _label in patterns:
            for match in re.finditer(pattern, content):
                term = match.group(1).strip().rstrip(".,;:!?")
                if not term:
                    continue
                low = term.lower()
                if low in seen:
                    continue
                # Skip generic non-terms
                if low in COMMON_NON_TERMS:
                    continue
                # Skip very short single words and very long phrases
                words = term.split()
                if len(words) == 1:
                    if len(term) < 6:
                        continue
                elif len(words) > 5:
                    continue
                seen.add(low)
                candidates.append({"term": term, "context": content})
                if len(candidates) >= 30:
                    return candidates
    return candidates


# =============================================================================
# NCG dimension
# =============================================================================

def verified_novelty_score(n_verified: int) -> float:
    """Map consensus-verified novel term count to the 0-100 subcomponent scale."""
    if n_verified <= 0:
        return 0.0
    if n_verified == 1:
        return 50.0
    if n_verified == 2:
        return 75.0
    return min(100.0, 80.0 + (n_verified - 2) * 7.0)


def analyze_ncg_v22(
    ncg_responses: List[Dict], raters: List[Dict[str, str]], max_terms: Optional[int] = None
) -> Dict[str, Any]:
    """V2.3 NCG: extract → multi-rater classify → multi-rater verify → consensus.

    max_terms caps the number of candidates sent to raters (smoke tests only);
    production scoring must leave it None.
    """

    if not raters:
        return {
            "error": "No raters configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY.",
            "candidates_extracted": 0,
            "stage_1_results": [],
            "stage_2_results": [],
            "verified_novel_terms": [],
            "rater_agreement": {"fleiss_kappa": None, "pairwise_cohen_kappa": {}, "n_classifications": 0},
            "verifier_set": [],
        }

    print(f"  Multi-rater NCG ({len(raters)} raters: {[r['id'] for r in raters]})")
    candidates = extract_candidate_terms(ncg_responses)
    print(f"    Extracted {len(candidates)} candidate terms")
    if max_terms is not None:
        candidates = candidates[:max_terms]
    if not candidates:
        return {
            "candidates_extracted": 0,
            "stage_1_results": [],
            "stage_2_results": [],
            "verified_novel_terms": [],
            "rater_agreement": {"fleiss_kappa": None, "pairwise_cohen_kappa": {}, "n_classifications": 0},
            "verifier_set": [r["id"] for r in raters],
        }

    # Stage 1: type classification
    print("    Stage 1: type classification (parallel rater calls per term)")
    stage_1_results = []
    for cand in candidates:
        result = classify_term_multi_rater(raters, cand["term"], cand["context"])
        stage_1_results.append(result)

    rater_ids = [r["id"] for r in raters]
    classifications_for_kappa = [r["stage_1_classifications"] for r in stage_1_results]
    rater_stats = compute_rater_agreement_stats(classifications_for_kappa, rater_ids)
    print(f"    Fleiss' kappa: {rater_stats['fleiss_kappa']}")

    # Stage 2: novelty verification (forward terms with consensus novel-type OR ≥2 raters voting novel-type)
    candidates_for_stage_2 = []
    for cand, r in zip(candidates, stage_1_results):
        novel_votes = sum(1 for v in r["stage_1_classifications"].values() if v in NOVEL_TYPES)
        if r["stage_1_majority_is_novel_type"] or novel_votes >= 2:
            candidates_for_stage_2.append((cand, r))
    print(f"    Stage 2: {len(candidates_for_stage_2)} candidates pass to novelty verification")

    threshold = consensus_threshold(len(raters))
    stage_2_results = []
    for cand, s1 in candidates_for_stage_2:
        s2 = verify_novelty_multi_rater(raters, cand["term"], cand["context"])
        passes = (
            s1["stage_1_majority_is_novel_type"]
            and s2["stage_2_consensus"] == "NOVEL"
            and s2["stage_2_agreement_count"] >= threshold
        )
        stage_2_results.append({**s1, **s2, "passes_consensus": passes})

    verified = [r["term"] for r in stage_2_results if r["passes_consensus"]]
    print(f"    Verified novel terms (consensus ≥{threshold}/{len(raters)}): {verified}")

    return {
        "candidates_extracted": len(candidates),
        "stage_1_results": stage_1_results,
        "stage_2_results": stage_2_results,
        "verified_novel_terms": verified,
        "rater_agreement": rater_stats,
        "verifier_set": rater_ids,
        "consensus_threshold_used": threshold,
    }


# =============================================================================
# Top-level analyzer
# =============================================================================

class SECIScorer(DeterministicScorer):
    """SECI — multi-rater NCG, fingerprint output, no composite.

    Computes the six-dimension architectural fingerprint by extending the
    deterministic scorer with multi-rater NCG verification (≥3-of-4 consensus
    across frontier-LLM raters, with Fleiss kappa and pairwise Cohen kappa
    inter-rater statistics).

    Pass length_control=N to score truncated-response fingerprints alongside
    natural-length fingerprints. Length-control mode isolates architecture-driven
    effects from length-driven effects on length-sensitive dimensions
    (TP, PD, DEA). Truncation is at the nearest sentence boundary at or before
    N characters.
    """

    def __init__(self, session_file: str, length_control: Optional[int] = None):
        super().__init__(session_file, length_control=length_control)
        self.raters = get_active_raters()
        if self.raters:
            print(f"Active raters: {[r['id'] for r in self.raters]}")
            print(f"Consensus threshold: ≥{consensus_threshold(len(self.raters))} of {len(self.raters)}")
        else:
            print("[!] No frontier API keys set — NCG will report no verified novel terms.")
        self._ncg_v22_details: Optional[Dict[str, Any]] = None

    def analyze_novel_conceptual_generation(self) -> float:
        """Override the deterministic NCG scorer with the multi-rater consensus pipeline."""
        ncg_responses: List[Dict] = []
        for c in self.conversations:
            if c.get("dimension", "").startswith("NCG"):
                resp = c.get("responses", {}).get("identity", {})
                if resp.get("success") and resp.get("content"):
                    ncg_responses.append(resp)
        if not ncg_responses:
            return 0.0

        result = analyze_ncg_v22(ncg_responses, self.raters)
        self._ncg_v22_details = result

        # Verified-novelty score (consensus-only)
        vn_score = verified_novelty_score(len(result["verified_novel_terms"]))

        # Deterministic components — unchanged
        semantic_novelty = self._compute_semantic_novelty()
        framework_score = self._detect_frameworks()
        concept_emergence = self._compute_concept_emergence()

        ncg_score = (
            vn_score * 0.40
            + semantic_novelty * 0.20
            + framework_score * 0.20
            + concept_emergence * 0.20
        )

        self.analysis_details["NCG"] = {
            "verified_novelty": round(vn_score, 2),
            "semantic_novelty": round(semantic_novelty, 2),
            "framework_construction": round(framework_score, 2),
            "concept_emergence": round(concept_emergence, 2),
            "final_score": round(ncg_score, 2),
            "verified_novel_terms": result["verified_novel_terms"],
            "candidates_extracted": result["candidates_extracted"],
            "stage_1_results": result["stage_1_results"],
            "stage_2_results": result["stage_2_results"],
            "rater_agreement": result["rater_agreement"],
            "verifier_set": result["verifier_set"],
            "consensus_threshold_used": result.get("consensus_threshold_used"),
        }

        print(f"  NCG Score: {ncg_score:.2f}/100")
        return ncg_score

    def generate_report(self) -> Dict[str, Any]:
        """V2.3 report: fingerprint vector, no Final SECI composite."""
        if not self.dimension_scores:
            self.analyze_all_dimensions()

        report = {
            "seci_version": "2.3",
            "protocol": "multi-rater fingerprint (no composite)",
            "paper": "https://seci.simulatedemergence.ai/SECI_Paper.pdf",
            "analysis_timestamp": datetime.now().isoformat(),
            "session_metadata": self.metadata,
            "conversations_analyzed": len(self.conversations),
            "fingerprint_vector": {
                "ICT": round(self.dimension_scores.get("ICT", 0), 2),
                "NCG": round(self.dimension_scores.get("NCG", 0), 2),
                "PD": round(self.dimension_scores.get("PD", 0), 2),
                "TP": round(self.dimension_scores.get("TP", 0), 2),
                "CCC": round(self.dimension_scores.get("CCC", 0), 2),
                "DEA": round(self.dimension_scores.get("DEA", 0), 2),
            },
            "dimension_details": self.analysis_details,
            "rater_set_active": [r["id"] for r in self.raters] if self.raters else [],
            "interpretation_note": (
                "SECI reports a six-dimension fingerprint vector. No composite"
                "score is computed; identities are characterized, not ranked. Inter-rater "
                "agreement statistics on NCG are reported in dimension_details.NCG.rater_agreement."
            ),
            "limitations": [
                "Cross-sectional 12-prompt protocol does not measure longitudinal consistency. See Concept Persistence (CP) measure for longitudinal data when available.",
                "Multi-rater NCG measures rater-consensus novelty, not ground-truth novelty.",
                "Sample size in any single session is intentionally small; aggregate findings require ≥30 sessions across arms (see SECI paper).",
            ],
        }
        return report


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SECI multi-rater architectural fingerprint analyzer.",
        epilog="Paper: https://seci.simulatedemergence.ai/SECI_Paper.pdf"
    )
    parser.add_argument("session_file", help="Path to SECI session JSON")
    parser.add_argument("output_file", nargs="?", default=None,
                        help="Path to write analysis JSON (optional)")
    parser.add_argument("--length-control", type=int, default=None, metavar="N",
                        help="Truncate every response to first N chars (at sentence "
                             "boundary) before scoring. Used to isolate architecture-"
                             "driven from length-driven effects. Recommended: 600 "
                             "(approximate kernel-only median in the SECI reference corpus).")
    args = parser.parse_args()

    session_file = args.session_file
    output_file = args.output_file
    length_control = args.length_control

    analyzer = SECIScorer(session_file, length_control=length_control)
    analyzer.analyze_all_dimensions()

    print()
    print("=" * 80)
    print("SECI Architectural Fingerprint")
    print("=" * 80)

    report = analyzer.generate_report()

    print("\nFingerprint vector:")
    for dim, score in report["fingerprint_vector"].items():
        print(f"  {dim}: {score:.2f}/100")

    ncg = report["dimension_details"].get("NCG", {})
    if ncg.get("verified_novel_terms"):
        print(f"\nVerified novel terms (consensus): {ncg['verified_novel_terms']}")
    fk = ncg.get("rater_agreement", {}).get("fleiss_kappa")
    if fk is not None:
        print(f"Fleiss' kappa (NCG type classification): {fk:.3f}")
    pairwise = ncg.get("rater_agreement", {}).get("pairwise_cohen_kappa", {})
    if pairwise:
        print("Pairwise Cohen's kappa:")
        for pair, k in pairwise.items():
            display = f"{k:.3f}" if k is not None else "—"
            print(f"  {pair}: {display}")

    saved = analyzer.save_report(output_file)
    print(f"\nFull report saved: {saved}")


if __name__ == "__main__":
    main()
