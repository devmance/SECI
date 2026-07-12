"""
LLMSubstrate — concrete IdentitySubstrate from SECI-format analysis JSONs.

Loads a SECI analysis result file (with `fingerprint_vector` + raw
conversations) into the substrate abstraction. This is the input format
the analysis layer consumes when working with multi-model multi-arm data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import IdentitySubstrate


_ARM_PATTERNS = {
    "A": ("arm_a", "arm_a_full_framework", "_arm_a_"),
    "B": ("arm_b", "_arm_b_"),
    "C": ("arm_c", "arm_c_kernel_only", "_arm_c_"),
}


def _infer_arm(meta: Dict[str, Any], filename: str) -> str:
    arm = meta.get("arm")
    if arm in {"A", "B", "C"}:
        return arm
    haystack = (
        (meta.get("session_name") or "").lower()
        + " "
        + filename.lower()
    )
    for label, tokens in _ARM_PATTERNS.items():
        if any(tok in haystack for tok in tokens):
            return label
    return "?"


def _extract_responses(d: Dict[str, Any]) -> List[str]:
    """Pull the identity response text from each conversation in order."""
    responses: List[str] = []
    for conv in d.get("conversations", []):
        r = conv.get("responses", {})
        identity_resp = r.get("identity") or r.get("response") or {}
        content = identity_resp.get("content") if isinstance(identity_resp, dict) else None
        if isinstance(content, str):
            responses.append(content)
        else:
            responses.append("")
    return responses


class LLMSubstrate(IdentitySubstrate):
    """
    LLM behavioral substrate parsed from a SECI analysis result file.

    Activity matrix is (T × N) where T = number of protocol turns and N is
    the sentence-embedding dimension. Embeddings are lazily computed (via
    sentence-transformers if available) only when `activity` is read; for
    re-analysis workflows that only need `dimension_scores` no embeddings
    are loaded.
    """

    def __init__(
        self,
        result_path: Path | str,
        embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self._path = Path(result_path)
        if not self._path.exists():
            raise FileNotFoundError(self._path)
        with self._path.open("r", encoding="utf-8") as fh:
            self._data = json.load(fh)
        self._embedding_model_name = embedding_model
        self._activity_cache: Optional[np.ndarray] = None
        self._responses_cache: Optional[List[str]] = None

    # --- Metadata --------------------------------------------------------

    @property
    def metadata(self) -> Dict[str, Any]:
        meta = self._data.get("session_metadata", {})
        protocol = meta.get("protocol_version") or self._data.get("protocol_version") or "v22"
        return {
            "model": meta.get("model", "?"),
            "identity": meta.get("identity_id", "?"),
            "identity_name": meta.get("identity_name", "?"),
            "identity_category": meta.get("identity_category", "?"),
            "arm": _infer_arm(meta, self._path.name),
            "protocol": protocol,
            "source_file": str(self._path),
        }

    # --- Dimension scores ------------------------------------------------

    @property
    def dimension_scores(self) -> Dict[str, float]:
        fp = self._data.get("fingerprint_vector")
        if fp is None:
            fp = self._data.get("dimension_scores", {})
        return {
            k: float(v) for k, v in fp.items()
            if k in {"ICT", "NCG", "PD", "TP", "CCC", "DEA"}
        }

    @property
    def dimension_details(self) -> Dict[str, Dict[str, Any]]:
        return self._data.get("dimension_details", {}) or {}

    # --- Raw conversation text + activity matrix -------------------------

    @property
    def responses(self) -> List[str]:
        if self._responses_cache is None:
            self._responses_cache = _extract_responses(self._data)
        return self._responses_cache

    @property
    def activity(self) -> np.ndarray:
        """
        Return (T × N) embedding matrix. Lazily computed; requires
        sentence-transformers. For analyses that only need dimension scores,
        do not read this property and the model never loads.
        """
        if self._activity_cache is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "Computing activity matrix requires sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                ) from e
            model = SentenceTransformer(self._embedding_model_name)
            embeddings = model.encode(self.responses, convert_to_numpy=True)
            self._activity_cache = embeddings.astype(np.float64)
        return self._activity_cache
