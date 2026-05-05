"""
llm_integration.py
------------------
LLM API integration layer. Supports OpenAI (default) and Anthropic.

Handles:
  - API client setup from environment variables
  - Rate-limit retries with exponential backoff
  - Structured JSON output parsing
  - Token usage tracking
  - Side-by-side uniqueness comparison
"""

import os
import time
from typing import Callable, Optional
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)


# ------------------------------------------------------------------ #
# RESPONSE DATACLASS
# ------------------------------------------------------------------ #

@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    metadata: dict = field(default_factory=dict)

    def __str__(self):
        return self.content


def _is_rate_limit_error(error: Exception) -> bool:
    """Return True for common provider rate limit errors."""
    error_str = str(error).lower()
    return "rate_limit" in error_str or "rate limit" in error_str or "too many requests" in error_str


def _retry_with_backoff(provider_name: str, max_retries: int, call_fn: Callable[[], LLMResponse]) -> LLMResponse:
    """Execute a provider call with exponential backoff on rate limits."""
    for attempt in range(1, max_retries + 1):
        try:
            return call_fn()
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < max_retries:
                wait = 2 ** attempt
                print(f"  {provider_name} rate limit hit. Waiting {wait}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"{provider_name} API error after {attempt} attempt(s): {e}") from e

    raise RuntimeError("Max retries exceeded.")


def _usage_value(usage, field_name: str) -> int:
    """Safely read a usage field returned by the API."""
    if not usage:
        return 0
    return getattr(usage, field_name, 0) or 0


def _trimmed_text(text: Optional[str]) -> str:
    """Normalize returned LLM text to a clean string."""
    return (text or "").strip()


# ------------------------------------------------------------------ #
# OPENAI CLIENT
# ------------------------------------------------------------------ #

class OpenAIClient:
    """Wrapper around OpenAI's chat completions API."""

    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 800,
        max_retries: int = 3,
        timeout: int = 60,
    ) -> LLMResponse:
        """Send a prompt and return an LLMResponse."""

        def _call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            usage = response.usage
            return LLMResponse(
                content=_trimmed_text(response.choices[0].message.content),
                model=self.model,
                prompt_tokens=_usage_value(usage, "prompt_tokens"),
                completion_tokens=_usage_value(usage, "completion_tokens"),
                total_tokens=_usage_value(usage, "total_tokens"),
            )

        return _retry_with_backoff("OpenAI", max_retries, _call)

    def generate_stream(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 800,
        timeout: int = 60,
    ):
        """Yield text chunks as they arrive (streaming)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ------------------------------------------------------------------ #
# ANTHROPIC CLIENT
# ------------------------------------------------------------------ #

class AnthropicClient:
    """Wrapper around Anthropic's Messages API."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Run: pip install anthropic")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 800,
        max_retries: int = 3,
        timeout: int = 60,
    ) -> LLMResponse:
        """Send a prompt and return an LLMResponse."""

        def _call():
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                timeout=timeout,
            )

            usage = response.usage
            content_blocks = getattr(response, "content", [])
            first_block = content_blocks[0] if content_blocks else None
            text = getattr(first_block, "text", "") if first_block else ""

            return LLMResponse(
                content=_trimmed_text(text),
                model=self.model,
                prompt_tokens=_usage_value(usage, "input_tokens"),
                completion_tokens=_usage_value(usage, "output_tokens"),
                total_tokens=_usage_value(usage, "input_tokens") + _usage_value(usage, "output_tokens"),
            )

        return _retry_with_backoff("Anthropic", max_retries, _call)

    def generate_stream(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 800,
        timeout: int = 60,
    ):
        """Yield text chunks as they arrive (streaming)."""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        ) as stream:
            for text in stream.text_stream:
                yield text


# ------------------------------------------------------------------ #
# FACTORY
# ------------------------------------------------------------------ #

def get_llm_client(provider: str = "openai") -> OpenAIClient | AnthropicClient:
    """
    Returns an LLM client based on provider string.
    Defaults to OpenAI if OPENAI_API_KEY is set, else Anthropic.
    """
    provider = provider.lower()

    if provider == "auto":
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            raise ValueError("Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY found in .env")

    if provider == "openai":
        return OpenAIClient()
    if provider == "anthropic":
        return AnthropicClient()
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")


# ------------------------------------------------------------------ #
# UNIQUENESS COMPARISON HELPER
# ------------------------------------------------------------------ #

def run_uniqueness_comparison(
    topic: str,
    fitbyte_prompt: str,
    fitbyte_system: str,
    llm: OpenAIClient | AnthropicClient,
) -> dict:
    """
    Generates content from both FitByte's branded prompt and a generic prompt,
    returning both for side-by-side comparison.

    Args:
        topic: The content topic
        fitbyte_prompt: Full branded user prompt
        fitbyte_system: FitByte system prompt with role definition
        llm: Initialized LLM client

    Returns:
        Dict with 'fitbyte_output', 'generic_output', and 'analysis'
    """
    from prompt_templates import build_generic_prompt

    print("  Generating FitByte branded content...")
    fitbyte_response = llm.generate(
        user_prompt=fitbyte_prompt,
        system_prompt=fitbyte_system,
        temperature=0.7,
    )

    print("  Generating generic baseline content...")
    generic_response = llm.generate(
        user_prompt=build_generic_prompt(topic),
        system_prompt="You are a helpful assistant.",
        temperature=0.7,
    )

    # Auto-analysis prompt
    analysis_prompt = f"""Compare these two blog posts about "{topic}" for a fitness watch brand.

POST A (branded system):
{fitbyte_response.content}

POST B (generic prompt):
{generic_response.content}

In 3-4 sentences, explain specifically how Post A differs from Post B in terms of:
1. Brand voice and tone distinctiveness
2. Specificity of data and product references
3. Audience relevance and contextual framing

Be concrete - quote specific phrases where relevant."""

    analysis = llm.generate(
        user_prompt=analysis_prompt,
        system_prompt="You are a content quality analyst.",
        temperature=0,
        max_tokens=300,
    )

    return {
        "topic": topic,
        "fitbyte_output": fitbyte_response.content,
        "generic_output": generic_response.content,
        "analysis": analysis.content,
        "fitbyte_tokens": fitbyte_response.total_tokens,
        "generic_tokens": generic_response.total_tokens,
    }
