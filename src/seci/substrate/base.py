"""
IdentitySubstrate abstraction.

Any system that produces identity-protocol responses qualifies as a substrate.
This abstraction lets the same analysis layer score:
  - Text-output substrates (LLM behavioral responses) — this module
  - Activation-level substrates (open-weight hidden states) — future work

Following the SEMCA-7 substrate-agnostic pattern: any system observable as a
(T × N) activity matrix can have the same dimension calculators applied to it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class IdentitySubstrate(ABC):
    """
    Abstract base for any identity-scaffolded system observable as activity.

    For text-output substrates the activity matrix is (T, N) where:
      T = number of protocol turns (12 in the SECI protocol)
      N = embedding dimension (384 for all-MiniLM-L6-v2)

    For activation-level substrates (Phase 2):
      T = generation timestep
      N = hidden-state dimension or attention-head count
    """

    @property
    @abstractmethod
    def activity(self) -> np.ndarray:
        """Return the (T × N) activity matrix."""

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        Return metadata describing this substrate. Required keys:
          model:       e.g. 'gemini-3-pro-preview'
          identity:    e.g. 'auren' (or 'base' for null-scaffold arm)
          arm:         'A' (full SE framework) | 'B' (base) | 'C' (kernel only)
          protocol:    e.g. 'v22_12prompt'
        """

    @property
    @abstractmethod
    def dimension_scores(self) -> Dict[str, float]:
        """
        Return the 6-dimension fingerprint vector.
        Keys: ICT, NCG, PD, TP, CCC, DEA — each in [0, 100].
        """

    def __repr__(self) -> str:  # pragma: no cover
        m = self.metadata
        return (
            f"IdentitySubstrate(model={m.get('model')}, "
            f"identity={m.get('identity')}, arm={m.get('arm')})"
        )
