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
import json
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


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
    ) -> LLMResponse:
        """Send a prompt and return an LLMResponse."""

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                usage = response.usage
                return LLMResponse(
                    content=response.choices[0].message.content.strip(),
                    model=self.model,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                )

            except Exception as e:
                error_str = str(e)
                if "rate_limit" in error_str.lower() and attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"  Rate limit hit. Waiting {wait}s (attempt {attempt}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"OpenAI API error after {attempt} attempt(s): {e}") from e

        raise RuntimeError("Max retries exceeded.")


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
    ) -> LLMResponse:
        """Send a prompt and return an LLMResponse."""

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=temperature,
                )

                usage = response.usage
                return LLMResponse(
                    content=response.content[0].text.strip(),
                    model=self.model,
                    prompt_tokens=usage.input_tokens if usage else 0,
                    completion_tokens=usage.output_tokens if usage else 0,
                    total_tokens=(usage.input_tokens + usage.output_tokens) if usage else 0,
                )

            except Exception as e:
                error_str = str(e)
                if "rate_limit" in error_str.lower() and attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"  Rate limit. Waiting {wait}s (attempt {attempt}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Anthropic API error: {e}") from e

        raise RuntimeError("Max retries exceeded.")


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
    elif provider == "anthropic":
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
    from prompt_templates import build_generic_prompt, SYSTEM_PROMPT_WRITER

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

Be concrete — quote specific phrases where relevant."""

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
