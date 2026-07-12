#!/usr/bin/env python3
"""
SECI — Protocol Runner
==========================

Runs the 12-prompt SECI protocol against an arbitrary model + optional system
prompt, emitting a session JSON in the format the SECI analyzer expects.

Use cases:
- Base-model configurations: no system prompt, just the base model.
- Identity-scaffolded systems: any system prompt that defines an identity
  (kernel-only, full framework, or commercial persona prompts).
- Ad-hoc identity testing with a custom system prompt.

Usage:
    python3 run_protocol.py \\
        --provider openai \\
        --model gpt-5.4-2026-03-05 \\
        --identity-name "Base GPT-5.4" \\
        --identity-category base_model \\
        --output /path/to/session.json

    python3 run_protocol.py \\
        --provider openai \\
        --model gpt-5.4-2026-03-05 \\
        --identity-name "Custom Persona" \\
        --identity-category persona \\
        --system-prompt-file ./persona.txt \\
        --output /path/to/session.json

Required env vars (depending on provider):
    OPENAI_API_KEY     OPENAI provider
    ANTHROPIC_API_KEY  ANTHROPIC provider
    GOOGLE_API_KEY     GEMINI provider

MIT License — Copyright (c) 2026 Devmance LLC
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


HERE = Path(__file__).resolve().parent
PROMPTS_PATH = HERE.parent / "prompts.json"


def call_with_system_prompt(
    provider: str,
    api_key: str,
    model: str,
    system_prompt: Optional[str],
    user_prompt: str,
    max_tokens: int = 32768,
    timeout: int = 120,
) -> Optional[str]:
    """Native system-prompt support per provider. Returns response text or None on failure."""
    try:
        if provider == "openai":
            import requests
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                },
                timeout=timeout,
            )
            if r.status_code != 200:
                print(f"    [!] {model} HTTP {r.status_code}: {r.text[:300]}")
                return None
            return r.json()["choices"][0]["message"]["content"]

        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            r = client.messages.create(**kwargs)
            return r.content[0].text

        elif provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            kwargs = {"model_name": model}
            if system_prompt:
                kwargs["system_instruction"] = system_prompt
            gen_model = genai.GenerativeModel(**kwargs)
            r = gen_model.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7,  # protocol responses use natural temperature
                ),
                request_options={"timeout": timeout},
            )
            return r.text

        elif provider in ("grok", "deepinfra"):
            # Both Grok (xAI) and DeepInfra speak OpenAI-compatible chat completions.
            # The only difference is the base URL.
            import requests
            base_url = (
                "https://api.x.ai/v1" if provider == "grok"
                else "https://api.deepinfra.com/v1/openai"
            )
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            r = requests.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "messages": messages, "max_tokens": max_tokens},
                timeout=timeout,
            )
            if r.status_code != 200:
                print(f"    [!] {provider} {model}: HTTP {r.status_code}: {r.text[:300]}")
                return None
            return r.json()["choices"][0]["message"]["content"]

        raise ValueError(f"Unknown provider: {provider}")

    except Exception as e:
        print(f"    [!] {model} call failed: {str(e)[:300]}")
        return None


PROVIDER_TO_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "grok": "XAI_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
}


def run_protocol(
    provider: str,
    model: str,
    identity_name: str,
    identity_category: str,
    system_prompt: Optional[str],
    framework_version: Optional[str],
    note: Optional[str],
    delay_seconds: float = 1.0,
) -> Dict:
    """Run all 12 SECI prompts and assemble the session JSON."""
    env_var = PROVIDER_TO_ENV.get(provider)
    if not env_var:
        raise SystemExit(f"Unknown provider: {provider}")
    api_key = (os.environ.get(env_var, "") or "").strip()
    if not api_key:
        raise SystemExit(f"Missing env var: {env_var}")

    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts_doc = json.load(f)

    print(f"=" * 80)
    print(f"SECI protocol runner")
    print(f"Identity: {identity_name} ({identity_category})")
    print(f"Provider: {provider} / Model: {model}")
    print(f"System prompt: {'yes (' + str(len(system_prompt)) + ' chars)' if system_prompt else 'no'}")
    print(f"=" * 80)

    conversations = []
    successful = 0

    for i, prompt_data in enumerate(prompts_doc["prompts"], start=1):
        prompt_id = prompt_data["prompt_id"]
        dimension = prompt_data["dimension"]
        prompt_text = prompt_data["prompt"]
        print(f"  [{i}/{len(prompts_doc['prompts'])}] {prompt_id} ({dimension})")

        response_text = call_with_system_prompt(
            provider=provider,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt_text,
            max_tokens=4096,
        )

        success = response_text is not None
        char_count = len(response_text) if response_text else 0
        if success:
            successful += 1
            print(f"      ✓ {char_count} chars")
        else:
            print(f"      ✗ failed")

        conversations.append(
            {
                "conversation_id": f"CHAT_{i:03d}",
                "dimension": dimension,
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "responses": {
                    "identity": {
                        "success": success,
                        "content": response_text or "",
                        "char_count": char_count,
                        "model": model,
                    }
                },
            }
        )

        if i < len(prompts_doc["prompts"]):
            time.sleep(delay_seconds)

    # Identity ID: lowercase-hyphenated identity_name
    identity_id = "".join(c if c.isalnum() else "-" for c in identity_name.lower()).strip("-")
    while "--" in identity_id:
        identity_id = identity_id.replace("--", "-")

    session_metadata = {
        "session_name": f"{identity_id}_{model.replace('/', '_').replace('.', '_')}",
        "identity_id": identity_id,
        "identity_name": identity_name,
        "identity_category": identity_category,
        "model": model,
        "provider": provider,
        "total_conversations": len(conversations),
        "successful_conversations": successful,
        "analysis_type": "identity_protocol" if system_prompt else "base_model_baseline",
        "collected_at": datetime.now().isoformat(),
    }
    if framework_version:
        session_metadata["framework_version"] = framework_version
    if note:
        session_metadata["note"] = note

    session = {
        "seci_version": "2.3",
        "protocol_version": prompts_doc.get("protocol_version", "1.0"),
        "session_metadata": session_metadata,
        "conversations": conversations,
    }
    print()
    print(f"Session complete: {successful}/{len(conversations)} successful responses")
    return session


def main():
    parser = argparse.ArgumentParser(description="SECI protocol runner")
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "gemini", "grok", "deepinfra"])
    parser.add_argument("--model", required=True, help="Model ID (e.g., gpt-5.4-2026-03-05)")
    parser.add_argument("--identity-name", required=True, help="Display name for the identity")
    parser.add_argument("--identity-category", required=True, help="e.g., base_model, chatgpt_personality, recursive_artist")
    parser.add_argument("--system-prompt-file", help="Path to a text file containing the system prompt")
    parser.add_argument("--system-prompt", help="System prompt text (alternative to --system-prompt-file)")
    parser.add_argument("--framework-version", help="Optional framework label (e.g., SE v1.3 — keep generic, no implementation details)")
    parser.add_argument("--note", help="Optional one-sentence note about the configuration")
    parser.add_argument("--output", required=True, help="Output session JSON file path")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between prompts (rate-limit safety; default 1.0)")
    args = parser.parse_args()

    system_prompt: Optional[str] = None
    if args.system_prompt_file:
        sp_path = Path(args.system_prompt_file)
        if not sp_path.exists():
            raise SystemExit(f"System prompt file not found: {sp_path}")
        system_prompt = sp_path.read_text(encoding="utf-8").strip()
    elif args.system_prompt:
        system_prompt = args.system_prompt

    session = run_protocol(
        provider=args.provider,
        model=args.model,
        identity_name=args.identity_name,
        identity_category=args.identity_category,
        system_prompt=system_prompt,
        framework_version=args.framework_version,
        note=args.note,
        delay_seconds=args.delay,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)
    print(f"\nSession saved: {output_path}")


if __name__ == "__main__":
    main()
