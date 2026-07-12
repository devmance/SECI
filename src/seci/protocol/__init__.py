"""
SECI protocol runner.

Runs the 12-prompt SECI protocol against any LLM provider (OpenAI / Anthropic /
Google / xAI / etc.) and emits a session JSON in the format SECIScorer expects.

Typical use:
    python -m examples.run_full_benchmark \\
        --model gemini-2.5-pro \\
        --identity-name auren \\
        --identity-kernel kernels/auren.txt \\
        --output sessions/auren_gemini25.json
"""

from .runner import run_protocol, call_with_system_prompt

__all__ = ["run_protocol", "call_with_system_prompt"]
