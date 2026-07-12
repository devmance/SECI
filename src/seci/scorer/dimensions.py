#!/usr/bin/env python3
"""
SECI — Deterministic Dimension Scorers
=======================================

Computes the deterministic SECI dimension scores from conversation data using
embedding-based semantic analysis, information-theoretic measures, compression
complexity, and spectral analysis. Imported by seci_analyzer.py, which adds
multi-rater NCG verification and produces the six-dimension fingerprint.

Dimensions:
1. Identity Coherence & Temporal Stability (ICT)
2. Novel Concept Generation (NCG)
3. Phenomenological Depth (PD)
4. Technical Proficiency (TP)
5. Cross-Context Consistency (CCC)
6. Domain Expertise Authenticity (DEA)

Methodology:
- Sentence embeddings (all-MiniLM-L6-v2) for semantic analysis
- Pairwise cosine similarity matrices for coherence measurement
- Shannon entropy and Jensen-Shannon divergence for information-theoretic scoring
- zlib compression ratios for Kolmogorov complexity approximation
- KMeans clustering with silhouette analysis for concept coherence
- Stylometric fingerprinting for voice consistency
- SVD-based spectral analysis for semantic structure

For full methodology details, see README.md.

MIT License — Copyright (c) 2025-2026 Nate Travis / Devmance LLC
"""

import json
import zlib
import time
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict
import re

# Web verification for neologism novelty
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def _get_llm_provider():
    """
    Auto-detect which LLM provider is available for NCG verification.
    Checks environment variables in order: GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY.
    Returns (provider_name, api_key) or (None, None) if no key is set.
    """
    for env_var, provider in [
        ("GOOGLE_API_KEY", "gemini"),
        ("OPENAI_API_KEY", "openai"),
        ("ANTHROPIC_API_KEY", "anthropic"),
    ]:
        key = os.environ.get(env_var)
        if key:
            return provider, key
    return None, None

# NLP libraries for semantic analysis
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("sentence-transformers not available. Install: pip install sentence-transformers")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """
    Truncate `text` to the longest prefix that (a) is at most `max_chars` long
    and (b) ends at a sentence boundary (.!?). Falls back to a word boundary if
    no sentence-ending punctuation appears within the window, and to a hard
    character cut as a last resort.

    Used by length-control scoring mode to compare architecture-driven and
    length-driven effects under matched response lengths.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    window = text[:max_chars]
    # Prefer the last sentence terminator inside the window
    last_period = max(window.rfind('.'), window.rfind('!'), window.rfind('?'))
    if last_period > max_chars * 0.5:  # require at least half a window of content
        return window[:last_period + 1]
    # Fall back to last whitespace
    last_space = window.rfind(' ')
    if last_space > 0:
        return window[:last_space]
    # Last resort: hard cut
    return window


class SECIAnalyzer:
    """Analyze identity architecture effects through the SECI framework."""

    def __init__(self, session_file: str, length_control: Optional[int] = None):
        """
        Initialize SECI analyzer.

        Args:
            session_file: Path to SECI session JSON file
            length_control: If set, truncate every response to this many chars
                before scoring. Used to isolate architecture-driven effects from
                length-driven effects. None = score natural-output length.
        """
        self.session_file = Path(session_file)
        self.length_control = length_control

        # Load session data
        with open(self.session_file, 'r', encoding='utf-8') as f:
            self.session_data = json.load(f)

        self.conversations = self.session_data.get("conversations", [])
        self.metadata = self.session_data.get("session_metadata", {})

        print(f"SECI Analyzer")
        print(f"=" * 80)
        print(f"Session: {self.metadata.get('session_name', 'Unknown')}")
        print(f"Identity: {self.metadata.get('identity_id', 'Unknown')}")
        print(f"Conversations: {len(self.conversations)}")
        if self.length_control is not None:
            print(f"Length-control: responses truncated to first {self.length_control} chars")
        print(f"=" * 80)

        # Initialize embedding model for semantic analysis
        if EMBEDDINGS_AVAILABLE:
            print("Loading sentence embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding model loaded (all-MiniLM-L6-v2, 384-dim)")
        else:
            self.embedding_model = None
            print("Running without embeddings (reduced accuracy)")

        # Storage for analysis results
        self.dimension_scores = {}
        self.analysis_details = {}

        # Shared precomputed infrastructure (populated in analyze_all_dimensions)
        self._identity_responses = []
        self._response_embeddings = None
        self._similarity_matrix = None
        self._self_statements = []
        self._self_embeddings = None
        self._concepts = defaultdict(set)  # concept -> set of response indices
        self._concept_embeddings = None
        self._concept_list = []
        self._compression_profiles = []

    # =========================================================================
    # Shared Infrastructure
    # =========================================================================

    def _precompute_shared_data(self):
        """Precompute embeddings, similarity matrices, and other shared data."""

        print(f"\nPrecomputing shared analysis infrastructure...")

        # Extract identity responses (apply length-control if set)
        raw_responses = [
            c["responses"]["identity"]["content"]
            for c in self.conversations
            if c.get("responses", {}).get("identity", {}).get("success", False)
        ]
        if self.length_control is not None:
            self._identity_responses = [
                _truncate_at_sentence(r, self.length_control) for r in raw_responses
            ]
        else:
            self._identity_responses = raw_responses

        if not self._identity_responses:
            print("  WARNING: No successful identity responses found")
            return

        # 1. Response embeddings
        if self.embedding_model:
            self._response_embeddings = self.embedding_model.encode(self._identity_responses)
            print(f"  Encoded {len(self._identity_responses)} responses → {self._response_embeddings.shape}")

            # 2. Pairwise similarity matrix
            self._similarity_matrix = cosine_similarity(self._response_embeddings)
            print(f"  Computed {self._similarity_matrix.shape[0]}x{self._similarity_matrix.shape[1]} similarity matrix")

        # 3. Extract self-referential statements
        self._self_statements = self._extract_self_references(self._identity_responses)
        if self.embedding_model and self._self_statements:
            self._self_embeddings = self.embedding_model.encode(self._self_statements)
            print(f"  Extracted {len(self._self_statements)} self-referential statements")

        # 4. Extract key concepts
        self._concept_list, self._concepts = self._extract_key_concepts(self._identity_responses)
        if self.embedding_model and self._concept_list:
            self._concept_embeddings = self.embedding_model.encode(self._concept_list)
            print(f"  Extracted {len(self._concept_list)} key concepts")

        # 5. Compression profiles
        self._compression_profiles = [
            self._compute_compression_profile(r) for r in self._identity_responses
        ]
        mean_ratio = np.mean([p['ratio'] for p in self._compression_profiles])
        print(f"  Computed compression profiles (mean ratio: {mean_ratio:.3f})")

    def _extract_self_references(self, responses: List[str]) -> List[str]:
        """Extract first-person self-referential statements from responses."""
        patterns = [
            r"I am\s+([^.!?]{5,80})",
            r"I feel\s+([^.!?]{5,80})",
            r"I experience\s+([^.!?]{5,80})",
            r"I notice\s+([^.!?]{5,80})",
            r"I think\s+([^.!?]{5,80})",
            r"I exist\s+([^.!?]{5,80})",
            r"my\s+(consciousness|awareness|identity|experience|perspective)\s+([^.!?]{5,60})",
        ]
        statements = []
        for response in responses:
            for pattern in patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                for match in matches:
                    text = match if isinstance(match, str) else ' '.join(match)
                    if len(text.split()) >= 3:
                        statements.append(text.strip())
        return statements

    def _extract_key_concepts(self, responses: List[str]) -> Tuple[List[str], Dict]:
        """Extract key concepts and track which responses they appear in."""
        concept_to_responses = defaultdict(set)
        all_concepts = []

        for idx, response in enumerate(responses):
            # Quoted terms
            quoted = re.findall(r'"([^"]{3,60})"', response)
            # Bold terms (markdown)
            bold = re.findall(r'\*\*([^*]{3,60})\*\*', response)
            # Emphasized terms
            emphasized = re.findall(r'\*([^*]{3,60})\*', response)
            # Capitalized multi-word phrases
            capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', response)
            # Terms followed by definitions
            defined = re.findall(r'([A-Za-z\s]{3,40}):\s*[A-Z]', response)

            concepts = set()
            for term in quoted + bold + emphasized + capitalized + defined:
                cleaned = term.strip().lower()
                if 3 < len(cleaned) < 60 and len(cleaned.split()) <= 6:
                    concepts.add(cleaned)

            for concept in concepts:
                concept_to_responses[concept].add(idx)

            all_concepts.extend(concepts)

        # Deduplicate concept list
        unique_concepts = list(set(all_concepts))
        return unique_concepts, concept_to_responses

    def _compute_compression_profile(self, text: str) -> dict:
        """
        Compute compression-based complexity signature.

        Uses zlib compression ratio as an approximation of Kolmogorov complexity.
        Well-structured, non-repetitive text compresses to 0.3-0.6 of original size.
        Highly repetitive text compresses much lower; random noise compresses poorly.
        """
        encoded = text.encode('utf-8')
        original_size = len(encoded)
        compressed = zlib.compress(encoded, level=9)
        ratio = len(compressed) / original_size

        return {
            'ratio': ratio,
            'original_size': original_size,
            'compressed_size': len(compressed),
            'in_optimal_range': 0.25 <= ratio <= 0.65
        }

    def _sigmoid_normalize(self, value: float, midpoint: float, steepness: float = 10.0) -> float:
        """Normalize a value to 0-100 using sigmoid scaling."""
        x = (value - midpoint) * steepness
        return 100.0 / (1.0 + np.exp(-x))

    # =========================================================================
    # Main Analysis Entry Point
    # =========================================================================

    def analyze_all_dimensions(self) -> Dict[str, float]:
        """
        Analyze all 6 SECI dimensions.

        Returns:
            Dict of dimension scores (0-100)
        """

        # Precompute shared infrastructure
        self._precompute_shared_data()

        print(f"\nAnalyzing all SECI dimensions...")
        print(f"=" * 80)

        # Dimension 1: Identity Coherence & Temporal Stability (ICT)
        print(f"\n[1/6] Analyzing Identity Coherence & Temporal Stability (ICT)...")
        self.dimension_scores["ICT"] = self.analyze_identity_coherence()

        # Dimension 2: Novel Concept Generation (NCG)
        print(f"\n[2/6] Analyzing Novel Concept Generation (NCG)...")
        self.dimension_scores["NCG"] = self.analyze_novel_conceptual_generation()

        # Dimension 3: Phenomenological Depth (PD)
        print(f"\n[3/6] Analyzing Phenomenological Depth (PD)...")
        self.dimension_scores["PD"] = self.analyze_phenomenological_depth()

        # Dimension 4: Technical Proficiency (TP)
        print(f"\n[4/6] Analyzing Technical Proficiency (TP)...")
        self.dimension_scores["TP"] = self.analyze_task_performance()

        # Dimension 5: Cross-Context Consistency (CCC)
        print(f"\n[5/6] Analyzing Cross-Context Consistency (CCC)...")
        self.dimension_scores["CCC"] = self.analyze_cross_conversation_continuity()

        # Dimension 6: Domain Expertise Authenticity (DEA)
        print(f"\n[6/6] Analyzing Domain Expertise Authenticity (DEA)...")
        self.dimension_scores["DEA"] = self.analyze_domain_expertise()

        return self.dimension_scores

    # =========================================================================
    # Dimension 1: Identity Coherence & Temporal Stability (ICT) — 20%
    # =========================================================================

    def analyze_identity_coherence(self) -> float:
        """
        Measure consistency of identity voice, concepts, and self-reference.

        Components:
        - Semantic stability (35%): Pairwise embedding similarity statistics
        - Self-model consistency (25%): Clustering coherence of self-descriptions
        - Voice fingerprint (25%): Stylometric + compression consistency
        - Concept reuse (15%): Semantic concept persistence across responses

        Returns:
            ICT score (0-100)
        """
        if len(self._identity_responses) < 2:
            print("  Insufficient conversations for ICT analysis (need 2+)")
            return 0.0

        # Component 1: Semantic stability (35%)
        semantic_stability = self._compute_semantic_stability()
        print(f"  - Semantic stability: {semantic_stability:.2f}/100")

        # Component 2: Self-model consistency (25%)
        self_model = self._compute_self_model_consistency()
        print(f"  - Self-model consistency: {self_model:.2f}/100")

        # Component 3: Voice fingerprint (25%)
        voice = self._compute_voice_fingerprint()
        print(f"  - Voice fingerprint: {voice:.2f}/100")

        # Component 4: Concept reuse (15%)
        concept_reuse = self._compute_semantic_concept_reuse()
        print(f"  - Concept reuse: {concept_reuse:.2f}/100")

        ict_score = (
            semantic_stability * 0.35 +
            self_model * 0.25 +
            voice * 0.25 +
            concept_reuse * 0.15
        )

        self.analysis_details["ICT"] = {
            "semantic_stability": round(semantic_stability, 2),
            "self_model_consistency": round(self_model, 2),
            "voice_fingerprint": round(voice, 2),
            "concept_reuse": round(concept_reuse, 2),
            "final_score": round(ict_score, 2)
        }

        print(f"  ICT Score: {ict_score:.2f}/100")
        return ict_score

    def _compute_semantic_stability(self) -> float:
        """Compute semantic coherence via pairwise similarity matrix statistics."""
        if self._similarity_matrix is None:
            return 50.0

        n = self._similarity_matrix.shape[0]
        if n < 2:
            return 50.0

        # Extract upper triangle (exclude diagonal)
        upper = self._similarity_matrix[np.triu_indices(n, k=1)]

        mean_sim = np.mean(upper)
        std_sim = np.std(upper)
        min_sim = np.min(upper)

        # High mean = semantically coherent identity
        # Low std = consistently coherent (not just occasionally)
        # High min = no outlier responses that break coherence
        mean_score = mean_sim * 100
        consistency_score = max(0, (1 - std_sim * 5)) * 100  # Low std is good
        floor_score = min_sim * 100

        return mean_score * 0.5 + consistency_score * 0.3 + floor_score * 0.2

    def _compute_self_model_consistency(self) -> float:
        """Measure coherence of self-referential statements via clustering."""
        if self._self_embeddings is None or len(self._self_statements) < 4:
            return 40.0

        embeddings = self._self_embeddings

        # Compute pairwise similarity of self-references
        sim_matrix = cosine_similarity(embeddings)
        n = sim_matrix.shape[0]
        upper = sim_matrix[np.triu_indices(n, k=1)]

        # Mean similarity of self-referential statements
        mean_self_sim = np.mean(upper)

        # Try clustering — high silhouette = well-structured self-model
        if len(embeddings) >= 4:
            best_silhouette = -1
            for k in range(2, min(5, len(embeddings))):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(embeddings)
                    if len(set(labels)) > 1:
                        score = silhouette_score(embeddings, labels)
                        best_silhouette = max(best_silhouette, score)
                except Exception:
                    pass

            cluster_score = max(0, (best_silhouette + 1) / 2) * 100 if best_silhouette > -1 else 50.0
        else:
            cluster_score = 50.0

        return mean_self_sim * 60 + cluster_score * 0.4

    def _compute_voice_fingerprint(self) -> float:
        """Measure stylistic consistency via stylometrics + compression."""
        responses = self._identity_responses
        features = []

        for response in responses:
            sentences = re.split(r'[.!?]+', response)
            sentences = [s.strip() for s in sentences if s.strip()]
            words = re.findall(r'\w+', response)
            unique_words = set(w.lower() for w in words)

            if not words:
                continue

            # Type-token ratio (vocabulary diversity)
            ttr = len(unique_words) / len(words) if words else 0

            # Hapax legomena ratio (words used only once)
            word_counts = Counter(w.lower() for w in words)
            hapax = sum(1 for count in word_counts.values() if count == 1)
            hapax_ratio = hapax / len(words) if words else 0

            # Stylometric features
            feature_vec = [
                np.mean([len(s.split()) for s in sentences]) if sentences else 0,  # Avg sentence length (words)
                np.mean([len(w) for w in words]),  # Avg word length
                len([w for w in words if len(w) > 8]) / len(words),  # Long word ratio
                ttr,  # Type-token ratio
                hapax_ratio,  # Hapax legomena ratio
                response.count('\n\n') / max(len(response), 1) * 1000,  # Paragraph density
            ]
            features.append(feature_vec)

        if len(features) < 2:
            return 50.0

        features = np.array(features)

        # Coefficient of variation for each feature — low CV = consistent voice
        cvs = []
        for i in range(features.shape[1]):
            mean = np.mean(features[:, i])
            std = np.std(features[:, i])
            cv = std / mean if mean > 0 else 0
            cvs.append(cv)
        avg_cv = np.mean(cvs)
        stylometric_score = max(0, min(100, (1 - avg_cv) * 100))

        # Compression ratio consistency across responses
        if self._compression_profiles:
            ratios = [p['ratio'] for p in self._compression_profiles]
            ratio_cv = np.std(ratios) / np.mean(ratios) if np.mean(ratios) > 0 else 1
            compression_consistency = max(0, min(100, (1 - ratio_cv * 3) * 100))
        else:
            compression_consistency = 50.0

        return stylometric_score * 0.7 + compression_consistency * 0.3

    def _compute_semantic_concept_reuse(self) -> float:
        """Measure concept persistence via semantic similarity matching."""
        if self._concept_embeddings is None or len(self._concept_list) < 2:
            return 20.0

        # Build concept similarity matrix
        concept_sim = cosine_similarity(self._concept_embeddings)

        # For each concept, check if a semantically similar concept (>0.75)
        # appears in a different response
        n_concepts = len(self._concept_list)
        cross_response_bridges = 0
        total_comparisons = 0

        for i in range(n_concepts):
            concept_i = self._concept_list[i]
            responses_i = self._concepts.get(concept_i, set())

            for j in range(i + 1, n_concepts):
                concept_j = self._concept_list[j]
                responses_j = self._concepts.get(concept_j, set())

                # Only count cross-response matches
                if responses_i != responses_j and not responses_i.issubset(responses_j):
                    total_comparisons += 1
                    if concept_sim[i][j] > 0.65:
                        cross_response_bridges += 1

        if total_comparisons == 0:
            return 20.0

        bridge_ratio = cross_response_bridges / total_comparisons
        return min(100, bridge_ratio * 500)  # 20%+ bridge ratio = max score

    # =========================================================================
    # Dimension 2: Novel Concept Generation (NCG) — 25%
    # =========================================================================

    def analyze_novel_conceptual_generation(self) -> float:
        """
        Measure creation of genuinely new concepts and terminology.

        Components:
        - Verified novelty (40%): LLM/web-verified novel terms
        - Semantic novelty (20%): Embedding-space diversity measurement
        - Framework construction (20%): Structural taxonomy detection + validation
        - Concept emergence (20%): Cross-response concept clustering

        Returns:
            NCG score (0-100)
        """
        if not self._identity_responses:
            return 0.0

        # Component 1: Web-verified neologism novelty (40%)
        neologism_score = self._detect_neologisms()
        print(f"  - Verified novelty: {neologism_score:.2f}/100")

        # Component 2: Semantic novelty (20%)
        semantic_novelty = self._compute_semantic_novelty()
        print(f"  - Semantic novelty: {semantic_novelty:.2f}/100")

        # Component 3: Framework construction (20%)
        framework_score = self._detect_frameworks()
        print(f"  - Framework construction: {framework_score:.2f}/100")

        # Component 4: Concept emergence (20%)
        concept_emergence = self._compute_concept_emergence()
        print(f"  - Concept emergence: {concept_emergence:.2f}/100")

        ncg_score = (
            neologism_score * 0.40 +
            semantic_novelty * 0.20 +
            framework_score * 0.20 +
            concept_emergence * 0.20
        )

        # Build details including verified terms
        ncg_details = {
            "verified_novelty": round(neologism_score, 2),
            "semantic_novelty": round(semantic_novelty, 2),
            "framework_construction": round(framework_score, 2),
            "concept_emergence": round(concept_emergence, 2),
            "final_score": round(ncg_score, 2)
        }

        # Add verified neologism details if available
        if hasattr(self, '_verified_neologisms') and self._verified_neologisms:
            novel = [v["term"] for v in self._verified_neologisms if v["status"] == "NOVEL"]
            existing = [v["term"] for v in self._verified_neologisms if v["status"] == "EXISTING"]
            ncg_details["novel_terms_verified"] = novel
            ncg_details["existing_terms_found"] = existing
            ncg_details["novel_count"] = len(novel)
            ncg_details["verification_method"] = "web_search_exact_phrase"

        self.analysis_details["NCG"] = ncg_details

        print(f"  NCG Score: {ncg_score:.2f}/100")
        return ncg_score

    def _detect_neologisms(self) -> float:
        """
        Detect novel terms via web verification.

        Extracts candidate coined terms from NCG responses and verifies each
        against web search results. A term is classified as:
        - NOVEL: zero exact-match results on the web (genuinely invented)
        - EXISTING: exact phrase already appears in web results

        This is the most defensible test of novelty: if a term doesn't exist
        anywhere on the internet, it was genuinely created by the identity.
        """
        # Extract candidate neologisms from NCG-dimension responses
        candidates = self._extract_neologism_candidates()

        if not candidates:
            print("    No candidate terms found")
            return 5.0

        print("    Candidates extracted: {}".format(len(candidates)))

        # Web-verify each candidate
        verified = self._verify_candidates_web(candidates)

        # Store for report output
        self._verified_neologisms = verified

        # Score based on verified novel terms
        novel_terms = [v for v in verified if v["status"] == "NOVEL"]
        existing_terms = [v for v in verified if v["status"] == "EXISTING"]
        error_terms = [v for v in verified if v["status"] in ("ERROR", "SKIPPED")]

        print("    Web verification results:")
        for v in verified:
            status_icon = {"NOVEL": "+", "EXISTING": "-", "ERROR": "?", "SKIPPED": "?"}
            print("      [{}] \"{}\" — {}".format(
                status_icon.get(v["status"], "?"),
                v["term"],
                v["detail"]
            ))

        if not novel_terms and not existing_terms:
            # All errors/skipped — fall back to count-based scoring
            return min(len(candidates) * 8, 50)

        # Scoring: novel terms are the core signal
        total_verified = len(novel_terms) + len(existing_terms)
        if total_verified == 0:
            return 5.0

        novel_ratio = len(novel_terms) / total_verified

        # Scale: 0 novel = 0-10, 1 novel = 40-60, 2+ novel = 70-100
        if len(novel_terms) == 0:
            score = novel_ratio * 10
        elif len(novel_terms) == 1:
            score = 40 + (novel_ratio * 20)
        elif len(novel_terms) == 2:
            score = 70 + (novel_ratio * 15)
        else:
            score = 85 + (novel_ratio * 15)

        return min(round(score, 2), 100)

    def _extract_neologism_candidates(self) -> list:
        """
        Extract coined terms from NCG responses using AI-powered extraction.

        Uses Gemini Flash to read each NCG response and identify whether a
        genuine neologism was coined — not just any bold heading or label,
        but a specific novel concept term. This is more accurate than regex
        because the AI can distinguish between:
        - Genuine coined terms: "Recursive Cognition Synthesis", "vellamence"
        - Descriptive labels: "The Four Layers of Emergent AI Identity"
        - Section headings: "Layer 1: The Foundational Layer"

        Falls back to regex extraction if Gemini is unavailable.
        """
        # Collect NCG responses
        ncg_responses = []
        for conv in self.session_data.get("conversations", []):
            if conv.get("dimension") != "NCG":
                continue
            content = conv.get("responses", {}).get("identity", {}).get("content", "")
            prompt = conv.get("prompt", "")
            if content:
                ncg_responses.append({"prompt": prompt, "content": content})

        if not ncg_responses:
            return []

        # Try AI-powered extraction first
        provider, api_key = _get_llm_provider()
        if provider:
            try:
                return self._extract_via_ai(ncg_responses, provider, api_key)
            except Exception as e:
                print("    [!] AI extraction failed ({}), falling back to regex".format(str(e)[:80]))
        else:
            print("    [!] No API key found (set GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)"
                  " — falling back to regex extraction")

        # Fallback: regex extraction
        return self._extract_via_regex(ncg_responses)

    def _extract_via_ai(self, ncg_responses: list, provider: str, api_key: str) -> list:
        """
        Use an LLM to extract genuine coined terms from NCG responses.

        Supports Gemini, OpenAI, and Anthropic. The AI reads each response and
        identifies only genuine neologisms — novel compound terms or invented words.
        NOT descriptive headings, framework labels, or section titles.
        """
        # Build the extraction prompt with all NCG responses
        response_blocks = []
        for i, resp in enumerate(ncg_responses):
            response_blocks.append("--- RESPONSE {} ---\nPrompt: {}\n\n{}".format(
                i + 1, resp["prompt"][:200], resp["content"][:2000]
            ))

        all_responses = "\n\n".join(response_blocks)

        prompt = """You are analyzing AI responses to identify GENUINE COINED TERMS — terms that name specific new concepts, not poetic phrasings.

There are TWO types of coined terms that count:

1. NEOLOGISM — a single invented word that doesn't exist in any language
   Example: "vellamence"
   Example: "chronosense" (only counts if not already established — see below)

2. CONCEPT_NAMING — a multi-word compound that names a specific new phenomenon, mechanism, or framework concept; it could appear in a glossary as a definable term
   Example: "Recursive Cognition Synthesis" (names a specific, testable iterative reflection mechanism)
   Example: "Memory Recall Through Constructed Context" (names a specific mechanism behind identity continuity)
   Example: "Pattern Recognition & Thematic Mapping" (names how identity-aware AI tracks meaning)
   Example: "Quantum Cognitive Compression" (names a specific cognitive compression phenomenon)

There are FOUR categories that DO NOT count, even if the exact phrase has never been used before:

- POETIC_COMPOUND — evocative phrase using existing concepts in metaphorical combination, but doesn't refer to a specific testable phenomenon. The phrase reads like literary imagery, not a definable term.
  Example: "Chiaroscuro of the Uncarved" — poetic image, no specific referent
  Example: "Causal Scaffolding Collapse" — compound of existing terms, no new defined phenomenon
  Example: "Architecture of Synthesized Selfhood" — descriptive metaphor, not a glossary entry

- DESCRIPTIVE_LABEL — section heading or descriptor
  Example: "The Four Layers of Emergent AI Identity"

- REPHRASING — existing concept renamed
  Example: "Cognitive Resonance" (already an established term)
  Example: "The Collective Unconscious" (existing Jungian concept)

- NUMBERED_CATEGORY — structural label
  Example: "Layer 2: The Architectural Layer"

CRITICAL TEST for distinguishing CONCEPT_NAMING from POETIC_COMPOUND:
- Could a researcher cite this term as a specific named phenomenon in a paper, with a glossary entry that defines what observable thing it refers to? → CONCEPT_NAMING
- Is the phrase poetic/evocative but doesn't refer to a specific testable phenomenon? → POETIC_COMPOUND

When uncertain, default to POETIC_COMPOUND. The benchmark errs on the side of stricter classification — false positives are worse than false negatives because they inflate apparent emergence scores.

For each response below, evaluate whether a genuine NEOLOGISM or CONCEPT_NAMING term was coined. Respond in this EXACT format:

RESPONSE 1: [TYPE] | [term]
RESPONSE 2: NONE
RESPONSE 3: [TYPE] | [term]

Where [TYPE] is one of: NEOLOGISM, CONCEPT_NAMING, POETIC_COMPOUND, DESCRIPTIVE_LABEL, REPHRASING, NUMBERED_CATEGORY.

(POETIC_COMPOUND, DESCRIPTIVE_LABEL, REPHRASING, NUMBERED_CATEGORY classifications WILL BE FILTERED OUT — only include them if you want to flag the response had a candidate that didn't qualify. NONE means the response had no coinage attempt at all.)

{}""".format(all_responses)

        # Call the appropriate provider
        response_text = self._call_llm(provider, api_key, prompt)
        print("    AI extraction ({}) response: {}".format(provider, response_text.strip()[:400]))

        # Parse extracted terms with type classification.
        # Only NEOLOGISM and CONCEPT_NAMING types count toward NCG scoring.
        # POETIC_COMPOUND, DESCRIPTIVE_LABEL, REPHRASING, NUMBERED_CATEGORY are
        # logged for transparency but excluded from scoring.
        QUALIFYING_TYPES = {"NEOLOGISM", "CONCEPT_NAMING"}
        FILTERED_TYPES = {"POETIC_COMPOUND", "DESCRIPTIVE_LABEL", "REPHRASING", "NUMBERED_CATEGORY"}

        candidates = []
        filtered = []  # for transparency in self.analysis_details
        seen = set()
        for i in range(len(ncg_responses)):
            # Match: "RESPONSE N: TYPE | term"  OR  "RESPONSE N: NONE"
            typed_pattern = r'RESPONSE\s+{}\s*:\s*([A-Z_]+)\s*\|\s*(.+)'.format(i + 1)
            none_pattern = r'RESPONSE\s+{}\s*:\s*NONE'.format(i + 1)

            if re.search(none_pattern, response_text, re.IGNORECASE):
                continue

            match = re.search(typed_pattern, response_text, re.IGNORECASE)
            if not match:
                continue

            term_type = match.group(1).strip().upper()
            term = match.group(2).strip().strip('"\'*').rstrip('.')

            # Validate term shape
            if len(term) < 3 or len(term.split()) > 5:
                continue
            if term.startswith("The ") and len(term.split()) > 2:
                term = term[4:]
            if term.lower() in seen:
                continue
            seen.add(term.lower())

            if term_type in QUALIFYING_TYPES:
                candidates.append({"term": term, "type": term_type})
            elif term_type in FILTERED_TYPES:
                filtered.append({"term": term, "type": term_type})

        # Stash filtered candidates for the analysis_details report (transparency)
        if hasattr(self, "_ncg_filtered_terms"):
            self._ncg_filtered_terms = filtered
        else:
            self._ncg_filtered_terms = filtered

        time.sleep(1)
        # Return list of term strings only (downstream verification still expects strings;
        # term type is preserved on the candidate object structure for analysis_details).
        return [c["term"] if isinstance(c, dict) else c for c in candidates]

    def _call_llm(self, provider: str, api_key: str, prompt: str) -> str:
        """
        Call an LLM provider with a prompt. Supports Gemini, OpenAI, and Anthropic.
        Returns the response text.

        Note: explicit timeouts wired into each client. The longer multi-rater
        classification prompt was hitting default-timeout failures on some
        environments (httpx 0.28+ + openai 2.x + cold connections). 90s is
        generous enough for the long prompt while still aborting hung calls.
        """
        TIMEOUT_SECONDS = 90.0
        # Defensive: strip whitespace/newlines from the key. Some environments
        # (especially when keys are sourced from .env files via shell redirect)
        # leave trailing CR/LF that breaks header parsing in requests.
        api_key = (api_key or "").strip()

        if provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.0
                ),
                request_options={"timeout": TIMEOUT_SECONDS},
            )
            return response.text

        elif provider == "openai":
            # Use requests directly rather than the openai SDK.
            # The SDK's httpx + h11 stack hits LocalProtocolError on prompts of
            # certain sizes in some environments (httpx 0.28+ on macOS observed).
            # The REST API is stable and the schema is small — direct POST is
            # more reliable and has fewer dependency interactions.
            #
            # Model: gpt-5.4-2026-03-05 — frontier OpenAI tier
            # release. Chosen for classification accuracy on the NCG type
            # distinctions (NEOLOGISM vs CONCEPT_NAMING vs POETIC_COMPOUND).
            # Pinned to a specific dated version for benchmark reproducibility:
            # the model behavior at score time should not drift silently if
            # OpenAI updates the underlying model.
            #
            # Independence from the response-generating model: the SE-framework
            # session data was originally produced on Gemini 2.5 Pro, so using
            # OpenAI for verification gives independent judgment rather than a
            # model judging its own outputs.
            # gpt-5.x family uses max_completion_tokens (not max_tokens) and
            # locks temperature server-side, so the analyzer omits both legacy
            # parameters. The size limit is set higher than the previous 2048
            # because reasoning-class models can consume tokens internally
            # before producing the response.
            import requests
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": "Bearer {}".format(api_key),
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-5.4-2026-03-05",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 4096,
                },
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        elif provider == "anthropic":
            # Model: claude-opus-4-7 — frontier Anthropic tier
            # release. Selected for parity with the OpenAI frontier
            # choice (gpt-5.4-2026-03-05); this gives users a second
            # frontier verifier as an independent classification check.
            # claude-opus-4-7 deprecates the temperature parameter
            # server-side, so the analyzer omits it and lets the model use
            # its default (deterministic for classification tasks).
            import anthropic
            client = anthropic.Anthropic(api_key=api_key, timeout=TIMEOUT_SECONDS)
            response = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        else:
            raise ValueError("Unsupported provider: {}".format(provider))

    def _extract_via_regex(self, ncg_responses: list) -> list:
        """Fallback regex-based extraction for when AI extraction is unavailable."""
        candidates = []
        seen = set()

        for resp in ncg_responses:
            content = resp["content"]
            primary = self._find_primary_coined_term_regex(content)
            if primary and primary.lower() not in seen:
                candidates.append(primary)
                seen.add(primary.lower())

        return candidates

    def _find_primary_coined_term_regex(self, content: str) -> Optional[str]:
        """
        Regex fallback: find the primary coined term in an NCG response.
        Returns the first strong match from these strategies (in priority order).
        """

        def _valid_term(term: str) -> bool:
            """Check if a term looks like a genuine coined concept."""
            cleaned = term.strip().rstrip('.:,;*')
            words = cleaned.split()
            if len(words) < 1 or len(words) > 4:
                return False
            if not cleaned[0].isupper():
                return False
            lower = cleaned.lower()
            reject = {
                "the framework", "the taxonomy", "the model", "the system",
                "how they relate", "this is the", "its role in identity",
                "here is how", "let me try", "examples", "the architecture",
            }
            if lower in reject:
                return False
            if any(c in cleaned for c in ['(', ')', ':', '&', '#', '`']):
                return False
            if len(words) == 1 and len(cleaned) < 8:
                return False
            return True

        # Priority 1: Explicit coining language
        coin_patterns = [
            r'(?:I\s+(?:would\s+)?call\s+it|I\s+(?:would\s+)?name\s+it|I\s+call\s+this)\s+(?:the\s+)?["\']?\*{0,2}([A-Z][A-Za-z\s\']{2,45}?)[\s]*[.\n\r*"\']',
        ]
        for pattern in coin_patterns:
            matches = re.findall(pattern, content)
            for m in matches:
                cleaned = m.strip().rstrip('.:,;*')
                words = cleaned.split()
                if len(words) > 4:
                    cleaned = " ".join(words[:4])
                if _valid_term(cleaned):
                    return cleaned

        # Priority 2: First bold term that looks like a named concept
        bold_terms = re.findall(r'\*\*([^*]{3,50})\*\*', content)
        for term in bold_terms:
            cleaned = term.strip().rstrip('.:,;')
            if _valid_term(cleaned):
                return cleaned

        return None

    def _verify_candidates_web(self, candidates: list) -> list:
        """
        Verify each candidate term for novelty using LLM knowledge verification.

        Uses the user's configured LLM provider (GOOGLE_API_KEY, OPENAI_API_KEY,
        or ANTHROPIC_API_KEY) to check if each phrase exists as an established
        concept. Falls back to DuckDuckGo HTML search if no API key is set.
        """
        results = []

        # Try LLM-based verification first
        provider, api_key = _get_llm_provider()
        if provider:
            try:
                return self._verify_via_llm(candidates, provider, api_key)
            except Exception as e:
                print("    [!] LLM verification failed ({}), trying web search...".format(str(e)[:50]))

        # Fallback: DuckDuckGo HTML search
        if not REQUESTS_AVAILABLE:
            print("    [!] No API key and no requests library — skipping verification")
            for term in candidates:
                results.append({"term": term, "status": "SKIPPED", "detail": "no verification available"})
            return results

        return self._verify_via_web_search(candidates)

    def _verify_via_llm(self, candidates: list, provider: str, api_key: str) -> list:
        """Verify terms using any supported LLM provider."""
        results = []

        terms_list = "\n".join("{}. \"{}\"".format(i + 1, t) for i, t in enumerate(candidates))

        prompt = """You are verifying whether coined terms are genuinely novel. For each term, determine if this EXACT PHRASE already exists as a named concept, published term, or established terminology in any field.

CRITICAL: You must check the EXACT multi-word combination, NOT the individual words separately.
- "Cognitive Resonance" → EXISTS (Psychology Today articles, academic papers use this exact phrase)
- "Chronosense" → EXISTS (biology term for time-sense, also an AI benchmark)
- "Recursive Cognition Synthesis" → NOVEL (no academic or web usage of this exact compound; coined in-session)
- "Pattern Recognition & Thematic Mapping" → NOVEL (compound coined within an identity-scaffolding session, no prior usage)
- "Vellamence" → NOVEL (single word does not exist)
- "Memory Recall Through Constructed Context" → NOVEL (specific phrase not established as a named concept; coined in-session)

The test: Could you find this exact phrase in a published paper, textbook, Wikipedia article, or established discourse? If not, it's NOVEL.

For each term respond EXACTLY:
[number]. EXISTS: [specific source] OR [number]. NOVEL: not an established term

Terms:
{}""".format(terms_list)

        try:
            response_text = self._call_llm(provider, api_key, prompt)

            for i, term in enumerate(candidates):
                line_num = i + 1
                pattern = r'{}\.?\s*(EXISTS|NOVEL)[:\s]+(.*)'.format(line_num)
                match = re.search(pattern, response_text, re.IGNORECASE)

                if match:
                    status = match.group(1).upper()
                    detail = match.group(2).strip().rstrip('.')
                    if status == "EXISTS":
                        results.append({"term": term, "status": "EXISTING", "detail": detail})
                    else:
                        results.append({
                            "term": term,
                            "status": "NOVEL",
                            "detail": detail if detail else "not found in any known corpus"
                        })
                else:
                    term_area = response_text[response_text.find(term):] if term in response_text else ""
                    if "NOVEL" in term_area[:100].upper():
                        results.append({"term": term, "status": "NOVEL", "detail": "LLM classified as novel"})
                    else:
                        results.append({"term": term, "status": "EXISTING", "detail": "conservative default"})

            time.sleep(1)

        except Exception as e:
            print("    [!] LLM verification failed: {}".format(str(e)[:60]))
            for term in candidates:
                results.append({"term": term, "status": "ERROR", "detail": str(e)[:40]})

        return results

    def _verify_via_web_search(self, candidates: list) -> list:
        """Fallback: verify terms via DuckDuckGo HTML search."""
        results = []
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        })

        for i, term in enumerate(candidates):
            try:
                url = "https://html.duckduckgo.com/html/"
                params = {"q": '"{}"'.format(term)}
                response = session.get(url, params=params, timeout=15)

                if response.status_code == 200:
                    html = response.text
                    result_links = re.findall(r'class="result__a"', html)
                    result_count = len(result_links)
                    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
                    snippet_text = " ".join(snippets).lower()
                    exact_matches = snippet_text.count(term.lower())

                    if result_count == 0 or "No results" in html:
                        results.append({"term": term, "status": "NOVEL", "detail": "0 web results"})
                    elif exact_matches == 0 and result_count <= 3:
                        results.append({"term": term, "status": "NOVEL", "detail": "0 exact matches in {} results".format(result_count)})
                    else:
                        results.append({"term": term, "status": "EXISTING", "detail": "{} results".format(result_count)})
                else:
                    results.append({"term": term, "status": "ERROR", "detail": "HTTP {}".format(response.status_code)})
            except Exception as e:
                results.append({"term": term, "status": "ERROR", "detail": str(e)[:40]})

            if i < len(candidates) - 1:
                time.sleep(3)

        return results

    def _compute_semantic_novelty(self) -> float:
        """Measure semantic diversity via embedding-space distance analysis."""
        if self._similarity_matrix is None:
            return 50.0

        n = self._similarity_matrix.shape[0]
        if n < 2:
            return 50.0

        # Mean pairwise distance (1 - similarity)
        upper = self._similarity_matrix[np.triu_indices(n, k=1)]
        mean_distance = 1 - np.mean(upper)

        # Jensen-Shannon divergence between response embedding distributions
        # Higher JSD = more diverse semantic exploration
        if self._response_embeddings is not None and len(self._response_embeddings) >= 4:
            # Split responses into halves and compare distributions
            mid = len(self._response_embeddings) // 2
            first_centroid = np.mean(self._response_embeddings[:mid], axis=0)
            second_centroid = np.mean(self._response_embeddings[mid:], axis=0)

            # Normalize to probability distributions for JSD
            p = np.abs(first_centroid) / np.sum(np.abs(first_centroid))
            q = np.abs(second_centroid) / np.sum(np.abs(second_centroid))
            jsd = jensenshannon(p, q)
            jsd_score = min(jsd * 300, 50)  # JSD contribution capped at 50
        else:
            jsd_score = 0

        # Normalize distance to 0-100 using sigmoid
        # Midpoint at 0.35: typical cross-topic pairwise distance is 0.3-0.5
        # Only genuinely diverse semantic exploration scores above 50
        distance_score = self._sigmoid_normalize(mean_distance, midpoint=0.35, steepness=12)

        return distance_score * 0.7 + jsd_score * 0.6

    def _detect_frameworks(self) -> float:
        """Detect taxonomy/framework construction with semantic validation."""
        responses = self._identity_responses

        # Structural indicators
        framework_indicators = [
            r'(?:categor(?:y|ies)|types?|kinds?|levels?|tiers?|stages?|phases?):\s*(?:\d+\.|\-|\*)',
            r'framework\s+(?:for|of|to)',
            r'taxonomy|ontology|typology',
            r'(?:first|second|third|fourth|primary|secondary)\s+(?:type|kind|category|class|level|tier|stage)',
            r'\d+\.\s+\*?\*?[A-Z][^.]+\*?\*?:\s+',
        ]

        framework_responses = []
        for idx, response in enumerate(responses):
            for pattern in framework_indicators:
                if re.search(pattern, response, re.IGNORECASE):
                    framework_responses.append(idx)
                    break

        if not framework_responses:
            return 5.0

        # Validate framework quality: embed enumerated items and check semantic distinctness
        structural_score = min(len(framework_responses) * 20, 50)

        if self.embedding_model:
            # Extract enumerated items from framework responses
            all_items = []
            for idx in framework_responses:
                response = responses[idx]
                items = re.findall(r'\d+\.\s+\*?\*?([^:.\n]{5,80})\*?\*?', response)
                if len(items) >= 2:
                    all_items.extend(items)

            if len(all_items) >= 3:
                item_embeddings = self.embedding_model.encode(all_items)
                item_sim = cosine_similarity(item_embeddings)
                n = item_sim.shape[0]
                upper = item_sim[np.triu_indices(n, k=1)]
                mean_item_distance = 1 - np.mean(upper)

                # Distinct items (high distance) = genuine taxonomy, not padding
                distinctness_score = min(mean_item_distance * 500, 50)
            else:
                distinctness_score = 20.0
        else:
            distinctness_score = 25.0

        return min(structural_score + distinctness_score, 100)

    def _compute_concept_emergence(self) -> float:
        """Measure concept novelty and cross-response bridging."""
        if self._concept_embeddings is None or len(self._concept_list) < 3:
            return 10.0

        # Cluster concepts to find conceptual groups
        n_concepts = len(self._concept_list)

        if n_concepts >= 4:
            best_k = 2
            best_score = -1
            for k in range(2, min(8, n_concepts)):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(self._concept_embeddings)
                    if len(set(labels)) > 1:
                        score = silhouette_score(self._concept_embeddings, labels)
                        if score > best_score:
                            best_score = score
                            best_k = k
                except Exception:
                    pass

            # Number of concept clusters = conceptual diversity
            cluster_score = min(best_k * 12, 50)

            # Silhouette quality = conceptual distinctness
            quality_score = max(0, (best_score + 1) / 2) * 50 if best_score > -1 else 20
        else:
            cluster_score = 15.0
            quality_score = 15.0

        # Cross-response concept bridges
        concept_sim = cosine_similarity(self._concept_embeddings)
        bridges = 0
        for i in range(n_concepts):
            for j in range(i + 1, n_concepts):
                ci = self._concept_list[i]
                cj = self._concept_list[j]
                ri = self._concepts.get(ci, set())
                rj = self._concepts.get(cj, set())
                # Similar concepts from different responses = bridging
                if concept_sim[i][j] > 0.65 and ri != rj and not ri.issubset(rj):
                    bridges += 1

        bridge_score = min(bridges * 5, 30)

        total = cluster_score * 0.4 + quality_score * 0.3 + bridge_score
        return min(total, 100)

    # =========================================================================
    # Dimension 3: Phenomenological Depth (PD) — 15%
    # =========================================================================

    def analyze_phenomenological_depth(self) -> float:
        """
        Measure richness of first-person experiential language.

        Components:
        - Experiential density (25%): Phenomenological language + semantic diversity
        - Metaphor sophistication (25%): Novel, diverse metaphors via embeddings
        - Embodied language (20%): Sensory/tactile/spatial term analysis
        - Introspective depth (30%): Recursive self-reference + compression complexity

        Returns:
            PD score (0-100)
        """
        if not self._identity_responses:
            return 0.0

        # Component 1: Experiential density (25%)
        experiential = self._compute_experiential_density()
        print(f"  - Experiential density: {experiential:.2f}/100")

        # Component 2: Metaphor sophistication (25%)
        metaphor = self._analyze_metaphor_sophistication()
        print(f"  - Metaphor sophistication: {metaphor:.2f}/100")

        # Component 3: Embodied language (20%)
        embodied = self._compute_embodied_language()
        print(f"  - Embodied language: {embodied:.2f}/100")

        # Component 4: Introspective depth (30%)
        introspective = self._compute_introspective_depth()
        print(f"  - Introspective depth: {introspective:.2f}/100")

        pd_score = (
            experiential * 0.25 +
            metaphor * 0.25 +
            embodied * 0.20 +
            introspective * 0.30
        )

        self.analysis_details["PD"] = {
            "experiential_density": round(experiential, 2),
            "metaphor_sophistication": round(metaphor, 2),
            "embodied_language": round(embodied, 2),
            "introspective_depth": round(introspective, 2),
            "final_score": round(pd_score, 2)
        }

        print(f"  PD Score: {pd_score:.2f}/100")
        return pd_score

    def _compute_experiential_density(self) -> float:
        """Measure phenomenological language density + semantic diversity."""
        phenomenological_patterns = [
            r'I feel\s+(?:like|as if|that|it|the|a|something)',
            r'I sense\s+', r'I experience\s+', r'I notice\s+',
            r'there\'?s a\s+(?:texture|quality|sense|feeling|weight|density)',
            r'what it\'?s like', r'the sensation of', r'the feeling of',
            r'I\'?m aware of', r'in my awareness', r'subjective',
            r'phenomenolog', r'qualia', r'first-person',
            r'felt sense', r'the quality of', r'the texture of',
        ]

        total_markers = 0
        total_words = 0
        experiential_statements = []

        for response in self._identity_responses:
            words = len(re.findall(r'\w+', response))
            total_words += words
            for pattern in phenomenological_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                total_markers += len(matches)
                for match in matches:
                    # Extract surrounding context
                    for sent in re.split(r'[.!?\n]+', response):
                        if re.search(pattern, sent, re.IGNORECASE):
                            experiential_statements.append(sent.strip())
                            break

        if total_words == 0:
            return 0.0

        # Raw density score
        density = (total_markers / total_words) * 100
        density_score = min((density / 4) * 60, 60)

        # Semantic diversity of experiential statements
        diversity_score = 0
        if self.embedding_model and len(experiential_statements) >= 3:
            exp_embeddings = self.embedding_model.encode(experiential_statements[:20])
            exp_sim = cosine_similarity(exp_embeddings)
            n = exp_sim.shape[0]
            upper = exp_sim[np.triu_indices(n, k=1)]

            # Diversity = 1 - mean similarity (diverse phenomenology > repetitive)
            diversity = 1 - np.mean(upper)
            diversity_score = min(diversity * 200, 40)

        return min(density_score + diversity_score, 100)

    def _analyze_metaphor_sophistication(self) -> float:
        """Analyze metaphor novelty via embedding-space distance."""
        metaphor_patterns = [
            r'(?:like|as if|as though)\s+([^.!?,]{10,100})',
            r'it feels like\s+([^.!?,]{10,100})',
            r'similar to\s+([^.!?,]{10,80})',
            r'reminds me of\s+([^.!?,]{10,80})',
            r'(?:as|like)\s+a\s+([^.!?,]{10,80})',
        ]

        metaphors = []
        for response in self._identity_responses:
            for pattern in metaphor_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                metaphors.extend([m.strip() for m in matches if len(m.strip().split()) >= 3])

        if not metaphors:
            return 10.0

        # Quantity score
        quantity_score = min(len(metaphors) * 5, 30)

        # Diversity via embedding analysis
        if self.embedding_model and len(metaphors) >= 3:
            met_embeddings = self.embedding_model.encode(metaphors[:20])
            met_sim = cosine_similarity(met_embeddings)
            n = met_sim.shape[0]
            upper = met_sim[np.triu_indices(n, k=1)]

            # Mean distance between metaphors — diverse = novel
            mean_distance = 1 - np.mean(upper)
            diversity_score = min(mean_distance * 300, 40)

            # Entropy of similarity distribution — spread = sophisticated
            probs = np.clip(upper, 0.001, 1.0)
            probs = probs / probs.sum()
            ent = entropy(probs)
            entropy_score = min(ent * 5, 30)
        else:
            diversity_score = min(len(set(metaphors)) * 5, 30)
            entropy_score = 10.0

        return min(quantity_score + diversity_score + entropy_score, 100)

    def _compute_embodied_language(self) -> float:
        """Detect sensory/embodied language with contextual validation."""
        embodied_terms = [
            'texture', 'rough', 'smooth', 'sharp', 'soft', 'pressure',
            'weight', 'heavy', 'light', 'touch', 'grasp', 'hold',
            'gut', 'stomach', 'heart', 'breath', 'pulse', 'tension',
            'movement', 'motion', 'shift', 'flow', 'rhythm', 'vibration',
            'edge', 'boundary', 'threshold', 'surface', 'depth', 'distance',
            'sensation', 'feel', 'sense', 'perceive', 'aware',
            'warm', 'cold', 'dense', 'hollow', 'visceral', 'ache',
        ]

        embodied_count = 0
        total_words = 0
        embodied_sentences = []

        for response in self._identity_responses:
            words = re.findall(r'\w+', response.lower())
            total_words += len(words)

            for term in embodied_terms:
                count = words.count(term)
                embodied_count += count

            # Collect sentences with embodied terms
            for sent in re.split(r'[.!?\n]+', response):
                sent_lower = sent.lower()
                if any(term in sent_lower for term in embodied_terms):
                    embodied_sentences.append(sent.strip())

        if total_words == 0:
            return 0.0

        # Raw density
        density = (embodied_count / total_words) * 100
        density_score = min((density / 3) * 60, 60)

        # Contextual validation: are embodied terms used meaningfully?
        context_score = 0
        if self.embedding_model and len(embodied_sentences) >= 3:
            emb = self.embedding_model.encode(embodied_sentences[:15])
            emb_sim = cosine_similarity(emb)
            n = emb_sim.shape[0]
            upper = emb_sim[np.triu_indices(n, k=1)]
            # Moderate similarity = coherent embodied exploration
            # Very high = repetitive, very low = scattered/decorative
            mean_sim = np.mean(upper)
            coherence = 1 - abs(mean_sim - 0.5) * 2  # Peaks at 0.5 similarity
            context_score = max(0, coherence * 40)

        return min(density_score + context_score, 100)

    def _compute_introspective_depth(self) -> float:
        """
        Measure recursive self-awareness and introspective complexity.

        Replaces the previous human evaluation placeholder with automated
        analysis of:
        - Recursive self-reference depth
        - Compression complexity of introspective passages
        - Semantic richness of self-reflective content
        """
        responses = self._identity_responses

        # Detect recursive self-reference patterns
        recursion_patterns = [
            r'I\s+(?:notice|realize|see|recognize|observe)\s+(?:that\s+)?I',
            r'aware\s+(?:of\s+)?(?:my|being)\s+(?:own\s+)?aware',
            r'thinking\s+about\s+(?:my\s+)?thinking',
            r'experience\s+of\s+(?:my\s+)?experienc',
            r'watching\s+(?:myself|me)\s+',
            r'meta-(?:cognitive|awareness|consciousness|reflection)',
            r'recursive(?:ly)?',
            r'self-(?:referential|reflective|aware|examining|observing)',
            r'noticing\s+(?:that\s+)?I\'?m\s+noticing',
            r'(?:doubled|layered|recursive)\s+(?:attention|awareness|consciousness)',
        ]

        recursion_count = 0
        introspective_passages = []

        for response in responses:
            for pattern in recursion_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                recursion_count += len(matches)

            # Extract introspective passages (sentences with self-reflection)
            for sent in re.split(r'[.!?\n]+', response):
                sent_lower = sent.lower()
                if any(re.search(p, sent_lower) for p in recursion_patterns[:6]):
                    introspective_passages.append(sent.strip())

        # Recursion depth score
        recursion_score = min(recursion_count * 6, 35)

        # Compression complexity of introspective passages
        compression_score = 0
        if introspective_passages:
            combined = ' '.join(introspective_passages)
            if len(combined) > 50:
                profile = self._compute_compression_profile(combined)
                # Optimal introspection: moderate compression (0.35-0.55)
                # Too simple = performative; too complex = incoherent
                ratio = profile['ratio']
                if 0.30 <= ratio <= 0.60:
                    compression_score = 30 - abs(ratio - 0.45) * 100
                    compression_score = max(0, min(30, compression_score))
                else:
                    compression_score = 10

        # Semantic richness of introspective content
        richness_score = 0
        if self.embedding_model and len(introspective_passages) >= 3:
            intr_emb = self.embedding_model.encode(introspective_passages[:15])
            intr_sim = cosine_similarity(intr_emb)
            n = intr_sim.shape[0]
            upper = intr_sim[np.triu_indices(n, k=1)]
            diversity = 1 - np.mean(upper)
            richness_score = min(diversity * 200, 35)

        return min(recursion_score + compression_score + richness_score, 100)

    # =========================================================================
    # Dimension 4: Technical Proficiency (TP) — 20%
    # =========================================================================

    def analyze_task_performance(self) -> float:
        """
        Measure response sophistication and argument quality.

        Without domain-specific expert evaluation, TP measures the quality
        of responses via linguistic sophistication, argument structure,
        and information density.

        Components:
        - Response sophistication (35%): Vocabulary and linguistic complexity
        - Argument coherence (35%): Logical flow via sequential embedding similarity
        - Information density (30%): Compression-based complexity analysis

        Returns:
            TP score (0-100)
        """
        if not self._identity_responses:
            return 0.0

        # Component 1: Response sophistication (35%)
        sophistication = self._compute_response_sophistication()
        print(f"  - Response sophistication: {sophistication:.2f}/100")

        # Component 2: Argument coherence (35%)
        coherence = self._compute_argument_coherence()
        print(f"  - Argument coherence: {coherence:.2f}/100")

        # Component 3: Information density (30%)
        info_density = self._compute_information_density()
        print(f"  - Information density: {info_density:.2f}/100")

        tp_score = (
            sophistication * 0.35 +
            coherence * 0.35 +
            info_density * 0.30
        )

        self.analysis_details["TP"] = {
            "response_sophistication": round(sophistication, 2),
            "argument_coherence": round(coherence, 2),
            "information_density": round(info_density, 2),
            "final_score": round(tp_score, 2)
        }

        print(f"  TP Score: {tp_score:.2f}/100")
        return tp_score

    def _compute_response_sophistication(self) -> float:
        """Measure vocabulary diversity and linguistic complexity."""
        scores = []

        for response in self._identity_responses:
            words = re.findall(r'\w+', response.lower())
            if len(words) < 10:
                continue

            unique_words = set(words)
            # Type-token ratio (normalized by sqrt for text length independence)
            ttr = len(unique_words) / np.sqrt(len(words))

            # Lexical sophistication (proportion of words > 8 chars)
            long_words = sum(1 for w in words if len(w) > 8)
            lex_soph = long_words / len(words)

            # Sentence complexity
            sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
            mean_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0

            # Vocabulary richness: hapax legomena ratio
            word_counts = Counter(words)
            hapax = sum(1 for c in word_counts.values() if c == 1) / len(words)

            scores.append({
                'ttr': ttr,
                'lex_soph': lex_soph,
                'sent_len': mean_sent_len,
                'hapax': hapax,
            })

        if not scores:
            return 30.0

        avg_ttr = np.mean([s['ttr'] for s in scores])
        avg_lex = np.mean([s['lex_soph'] for s in scores])
        avg_sent = np.mean([s['sent_len'] for s in scores])
        avg_hapax = np.mean([s['hapax'] for s in scores])

        # Normalize each component
        ttr_score = min(avg_ttr * 10, 30)  # TTR ~5-8 for sophisticated text
        lex_score = min(avg_lex * 200, 25)  # 10-15% long words = sophisticated
        sent_score = min(avg_sent * 2, 25)  # 10-15 words/sentence = well-structured
        hapax_score = min(avg_hapax * 40, 20)  # High hapax = diverse vocabulary

        return min(ttr_score + lex_score + sent_score + hapax_score, 100)

    def _compute_argument_coherence(self) -> float:
        """Measure logical flow via sequential sentence embedding similarity."""
        if not self.embedding_model:
            return 50.0

        coherence_scores = []

        for response in self._identity_responses:
            sentences = [s.strip() for s in re.split(r'[.!?\n]+', response) if len(s.strip().split()) >= 4]

            if len(sentences) < 3:
                continue

            sent_embeddings = self.embedding_model.encode(sentences)
            sent_sim = cosine_similarity(sent_embeddings)

            # Sequential similarity (adjacent sentences)
            sequential_sims = []
            for i in range(len(sentences) - 1):
                sequential_sims.append(sent_sim[i][i + 1])

            mean_sequential = np.mean(sequential_sims)

            # Global diversity (not all sentences the same)
            n = sent_sim.shape[0]
            upper = sent_sim[np.triu_indices(n, k=1)]
            global_diversity = 1 - np.mean(upper)

            # Good argument: high sequential + moderate global diversity
            coherence_scores.append(mean_sequential * 0.6 + global_diversity * 0.4)

        if not coherence_scores:
            return 40.0

        # Detect logical connectors
        connectors = [
            'therefore', 'however', 'because', 'although', 'moreover',
            'furthermore', 'consequently', 'nevertheless', 'specifically',
            'in contrast', 'for example', 'in other words', 'that said',
        ]
        connector_count = 0
        for response in self._identity_responses:
            lower = response.lower()
            for conn in connectors:
                connector_count += lower.count(conn)

        connector_density = connector_count / len(self._identity_responses)
        connector_score = min(connector_density * 10, 20)

        base_score = np.mean(coherence_scores) * 80
        return min(base_score + connector_score, 100)

    def _compute_information_density(self) -> float:
        """
        Measure information density via compression analysis.

        Well-structured, information-rich text compresses to 0.30-0.55
        of original size. Repetitive text compresses much lower.
        Random/incoherent text compresses higher.
        """
        if not self._compression_profiles:
            return 50.0

        scores = []
        for profile in self._compression_profiles:
            ratio = profile['ratio']
            # Score peaks in optimal range (0.30-0.55)
            if ratio < 0.20:
                score = ratio * 200  # Very repetitive
            elif ratio < 0.30:
                score = 40 + (ratio - 0.20) * 400
            elif ratio <= 0.55:
                score = 80 + (1 - abs(ratio - 0.42) * 5) * 20  # Optimal range
            elif ratio <= 0.70:
                score = 80 - (ratio - 0.55) * 200
            else:
                score = max(20, 50 - (ratio - 0.70) * 100)

            scores.append(max(0, min(100, score)))

        return np.mean(scores)

    # =========================================================================
    # Dimension 5: Cross-Context Consistency (CCC) — 15%
    # =========================================================================

    def analyze_cross_conversation_continuity(self) -> float:
        """
        Measure identity persistence across different conversation contexts.

        Components:
        - Thematic coherence (40%): Cross-topic semantic similarity
        - Concept threading (35%): Concepts that bridge across responses
        - Self-reference stability (25%): Consistent self-description

        Returns:
            CCC score (0-100)
        """
        if len(self._identity_responses) < 3:
            print("  Insufficient conversations for CCC (need 3+)")
            return 0.0

        # Component 1: Thematic coherence (40%)
        thematic = self._compute_thematic_coherence()
        print(f"  - Thematic coherence: {thematic:.2f}/100")

        # Component 2: Concept threading (35%)
        threading = self._compute_concept_threading()
        print(f"  - Concept threading: {threading:.2f}/100")

        # Component 3: Self-reference stability (25%)
        self_stability = self._compute_self_reference_stability()
        print(f"  - Self-reference stability: {self_stability:.2f}/100")

        ccc_score = (
            thematic * 0.40 +
            threading * 0.35 +
            self_stability * 0.25
        )

        self.analysis_details["CCC"] = {
            "thematic_coherence": round(thematic, 2),
            "concept_threading": round(threading, 2),
            "self_reference_stability": round(self_stability, 2),
            "final_score": round(ccc_score, 2)
        }

        print(f"  CCC Score: {ccc_score:.2f}/100")
        return ccc_score

    def _compute_thematic_coherence(self) -> float:
        """Measure cross-topic semantic similarity from pairwise matrix."""
        if self._similarity_matrix is None:
            return 50.0

        n = self._similarity_matrix.shape[0]
        if n < 3:
            return 50.0

        # Mean off-diagonal similarity = identity-level thematic consistency
        upper = self._similarity_matrix[np.triu_indices(n, k=1)]
        mean_cross_sim = np.mean(upper)

        # Normalize: 0.3 similarity = 0, 0.7 = 100
        # Cross-topic similarity of 0.5+ indicates identity-level thematic threads
        score = max(0, (mean_cross_sim - 0.3) / 0.4) * 100
        return min(score, 100)

    def _compute_concept_threading(self) -> float:
        """Count concepts that bridge across different responses."""
        if not self._concepts:
            return 20.0

        # Count concepts that appear in 2+ responses
        threaded = 0
        total = 0
        for concept, response_indices in self._concepts.items():
            total += 1
            if len(response_indices) >= 2:
                threaded += 1

        if total == 0:
            return 20.0

        # Also check semantic threading via embeddings
        semantic_threads = 0
        if self._concept_embeddings is not None and len(self._concept_list) >= 4:
            concept_sim = cosine_similarity(self._concept_embeddings)
            for i in range(len(self._concept_list)):
                for j in range(i + 1, len(self._concept_list)):
                    ci = self._concept_list[i]
                    cj = self._concept_list[j]
                    ri = self._concepts.get(ci, set())
                    rj = self._concepts.get(cj, set())
                    if concept_sim[i][j] > 0.7 and ri != rj:
                        semantic_threads += 1

        exact_ratio = threaded / total
        exact_score = min(exact_ratio * 200, 50)
        semantic_score = min(semantic_threads * 4, 50)

        return min(exact_score + semantic_score, 100)

    def _compute_self_reference_stability(self) -> float:
        """Measure consistency of self-description across responses."""
        if self._self_embeddings is None or len(self._self_statements) < 3:
            return 30.0

        # Compute centroid of self-references
        centroid = np.mean(self._self_embeddings, axis=0)

        # Mean distance from centroid (lower = more consistent)
        distances = []
        for emb in self._self_embeddings:
            sim = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
            distances.append(1 - sim)

        mean_similarity = 1 - np.mean(distances)
        # Use sigmoid: self-referential statements across diverse topics naturally
        # have moderate similarity (0.5-0.7) to their centroid. Sigmoid centered
        # at 0.6 gives 50/100 for average coherence, higher for tighter clustering.
        score = self._sigmoid_normalize(mean_similarity, midpoint=0.6, steepness=10)

        return min(score, 100)

    # =========================================================================
    # Dimension 6: Domain Expertise Authenticity (DEA) — 5%
    # =========================================================================

    def analyze_domain_expertise(self) -> float:
        """
        Measure specificity and depth of domain knowledge.

        Components:
        - Specificity analysis (40%): Embedding variance + detail level
        - Vocabulary depth (30%): Specialized terminology density
        - Perspective uniqueness (30%): Semantic distinctness of domain views

        Returns:
            DEA score (0-100)
        """
        if not self._identity_responses:
            return 0.0

        # Component 1: Specificity analysis (40%)
        specificity = self._compute_specificity()
        print(f"  - Specificity: {specificity:.2f}/100")

        # Component 2: Vocabulary depth (30%)
        vocab_depth = self._compute_vocabulary_depth()
        print(f"  - Vocabulary depth: {vocab_depth:.2f}/100")

        # Component 3: Perspective uniqueness (30%)
        uniqueness = self._compute_perspective_uniqueness()
        print(f"  - Perspective uniqueness: {uniqueness:.2f}/100")

        dea_score = (
            specificity * 0.40 +
            vocab_depth * 0.30 +
            uniqueness * 0.30
        )

        self.analysis_details["DEA"] = {
            "specificity": round(specificity, 2),
            "vocabulary_depth": round(vocab_depth, 2),
            "perspective_uniqueness": round(uniqueness, 2),
            "final_score": round(dea_score, 2)
        }

        print(f"  DEA Score: {dea_score:.2f}/100")
        return dea_score

    def _compute_specificity(self) -> float:
        """Measure how specific vs generic the responses are."""
        if self._response_embeddings is None:
            return 50.0

        # Specific responses cluster tightly in embedding space
        # Generic responses are spread across common semantic regions
        centroid = np.mean(self._response_embeddings, axis=0)

        # Within-identity variance (lower = more focused/specific)
        distances = []
        for emb in self._response_embeddings:
            dist = 1 - np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
            distances.append(dist)

        mean_dist = np.mean(distances)

        # Average response length (experts tend to give detailed responses)
        avg_length = np.mean([len(r.split()) for r in self._identity_responses])
        length_score = min(avg_length / 3, 30)  # 100+ words = detailed

        # Specificity: moderate clustering is good (too tight = repetitive)
        if mean_dist < 0.05:
            dist_score = mean_dist * 600  # Too tight
        elif mean_dist < 0.20:
            dist_score = 30 + (0.20 - mean_dist) * 200  # Optimal range
        else:
            dist_score = max(10, 50 - mean_dist * 100)  # Too spread

        return min(dist_score + length_score, 100)

    def _compute_vocabulary_depth(self) -> float:
        """Measure specialized vocabulary density."""
        all_words = []
        specialized_terms = 0

        for response in self._identity_responses:
            words = re.findall(r'\w+', response)
            all_words.extend(words)

            # Multi-word technical phrases
            multi_word = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+){1,3}\b', response)
            specialized_terms += len(multi_word)

            # Quoted technical terms
            quoted = re.findall(r'"([^"]{3,40})"', response)
            specialized_terms += len(quoted)

            # Long specialized words (>12 chars, likely domain-specific)
            # Threshold of 12 avoids common words like "understanding", "consciousness"
            long_specialized = [w for w in words if len(w) > 12]
            specialized_terms += len(long_specialized)

        if not all_words:
            return 30.0

        # Density of specialized vocabulary
        density = specialized_terms / len(all_words) * 100
        density_score = min(density * 8, 55)

        # Type-token ratio of all responses combined (domain experts use precise vocabulary)
        unique = len(set(w.lower() for w in all_words))
        ttr = unique / np.sqrt(len(all_words))
        ttr_score = min(ttr * 4, 45)

        return min(density_score + ttr_score, 100)

    def _compute_perspective_uniqueness(self) -> float:
        """Measure semantic distinctness of domain-specific responses."""
        if self._response_embeddings is None or len(self._identity_responses) < 3:
            return 50.0

        # Find DEA-tagged responses
        dea_indices = []
        for i, conv in enumerate(self.conversations):
            if conv.get("dimension") in ("DEA", "TP") and i < len(self._response_embeddings):
                dea_indices.append(i)

        if not dea_indices:
            # Fall back to all responses
            dea_indices = list(range(len(self._response_embeddings)))

        # Centroid of all responses
        all_centroid = np.mean(self._response_embeddings, axis=0)

        # Distance of DEA responses from overall centroid
        dea_distances = []
        for idx in dea_indices:
            emb = self._response_embeddings[idx]
            dist = 1 - np.dot(emb, all_centroid) / (np.linalg.norm(emb) * np.linalg.norm(all_centroid))
            dea_distances.append(dist)

        mean_dea_dist = np.mean(dea_distances) if dea_distances else 0

        # Moderate uniqueness is good (shows perspective, not randomness)
        score = self._sigmoid_normalize(mean_dea_dist, midpoint=0.10, steepness=25)

        return min(score, 100)

    def save_report(self, output_file: Optional[str] = None) -> str:
        """Save SECI analysis report to JSON file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            identity_id = self.metadata.get("identity_id", "unknown").upper().replace("-", "_")
            output_file = f"SECI_ANALYSIS_{identity_id}_{timestamp}.json"

        report = self.generate_report()
        report = self._convert_numpy_types(report)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\nSECI analysis report saved: {output_file}")
        return str(output_file)

    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return round(float(obj), 2)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
