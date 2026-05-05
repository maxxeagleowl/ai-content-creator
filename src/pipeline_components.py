"""
pipeline_components.py
----------------------
Small pipeline building blocks used by ContentPipeline.

These classes keep the orchestration layer thin and make it easier to test
individual stages without running the whole CLI/UI flow.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from knowledge_base import KnowledgeBase
from llm_integration import run_uniqueness_comparison
from prompt_templates import (
    PromptContext,
    build_analyst_system_prompt,
    build_brief_prompt,
    build_writer_system_prompt,
    get_prompt_for_channel,
)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _parse_json_response(text: str) -> dict:
    """Parse JSON from a model response, with fenced-block fallback."""
    raw = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_analysis": text}


def _slugify(value: str, max_length: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower().strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:max_length] or "content"


def build_prompt_context(
    topic: str,
    channel: str,
    audience: str,
    context: dict,
    *,
    extra_instructions: str = "",
) -> PromptContext:
    """Create a PromptContext from a knowledge-base context dict."""
    return PromptContext(
        topic=topic,
        channel=channel,
        audience=audience,
        brand_identity=context.get("brand_identity", ""),
        brand_voice=context.get("brand_voice", ""),
        writing_rules=context.get("writing_rules", ""),
        content_examples=context.get("content_examples", ""),
        product_specs=context.get("product_specs", ""),
        market_context=context.get("market_context", ""),
        differentiators=context.get("differentiators", ""),
        audience_insights=context.get("audience_insights", ""),
        extra_instructions=extra_instructions or "",
    )


@dataclass
class ContentDocumenter:
    kb: KnowledgeBase

    def document(self, topic: str, audience: str) -> dict:
        """Load knowledge base context with error handling."""
        try:
            if not topic or not topic.strip():
                raise ValueError("Topic cannot be empty")
            
            context = self.kb.full_context_for_generation(topic=topic, audience=audience)
            
            # Validate that we got meaningful context
            if not context:
                raise RuntimeError("Knowledge base returned empty context")
            
            for key, value in context.items():
                if not isinstance(value, str):
                    raise TypeError(f"Expected string for {key}, got {type(value).__name__}")
            
            return context
        except FileNotFoundError as e:
            raise RuntimeError(f"Knowledge base file missing: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load knowledge base: {e}")


@dataclass
class ContentMonitor:
    llm: object
    logger: Callable[[str], None]

    def analyze(self, topic: str, context: dict) -> dict:
        """Analyze topic for brand fit with error handling."""
        try:
            if not topic or not topic.strip():
                raise ValueError("Topic cannot be empty")
            
            if not context or not isinstance(context, dict):
                raise ValueError("Invalid context provided")
            
            analysis_prompt = f"""You are a content strategist for FitByte, a precision fitness watch brand.

Analyze this content topic: "{topic}"

Use this market context:
{context.get('market_context', '')[:800]}

FitByte brand identity:
{context.get('brand_identity', '')[:600]}

Answer briefly:
1. BRAND FIT (1-10): How well does this topic align with FitByte's voice and audience?
2. MARKET RELEVANCE: Which consumer trend does this address?
3. BEST ANGLE: What's the most distinctive angle for FitByte specifically?
4. RISK: Any brand voice pitfalls to avoid for this topic?

Return as JSON with keys: brand_fit_score, market_relevance, best_angle, risk"""

            response = self.llm.generate(
                user_prompt=analysis_prompt,
                system_prompt=build_analyst_system_prompt(
                    PromptContext(topic=topic, audience="general")
                ),
                temperature=0,
                max_tokens=400,
            )
            
            if not response or not response.content.strip():
                raise RuntimeError("LLM returned empty analysis")

            report = _parse_json_response(response.content)
            self.logger(f"  OK Brand fit score: {report.get('brand_fit_score', 'N/A')}/10")
            best_angle = str(report.get("best_angle", "N/A"))
            self.logger(f"  OK Best angle: {best_angle[:80]}")
            return report
        except Exception as e:
            self.logger(f"  ⚠ Analysis failed: {str(e)[:80]}")
            return {"brand_fit_score": 0, "error": str(e)}


@dataclass
class ContentBriefGenerator:
    llm: object
    logger: Callable[[str], None]

    def generate(self, topic: str, channel: str, audience: str, context: dict) -> str:
        """Generate content brief with error handling."""
        try:
            if not topic or not topic.strip():
                raise ValueError("Topic cannot be empty")
            if not context:
                raise ValueError("Context cannot be empty")
            
            ctx = build_prompt_context(topic, channel, audience, context)
            brief_response = self.llm.generate(
                user_prompt=build_brief_prompt(ctx),
                system_prompt=build_analyst_system_prompt(ctx),
                temperature=0.5,
                max_tokens=600,
            )
            
            if not brief_response or not brief_response.content.strip():
                raise RuntimeError("LLM failed to generate brief")

            self.logger(f"  OK Brief generated ({brief_response.total_tokens} tokens used)")
            return brief_response.content
        except Exception as e:
            self.logger(f"  ✗ Brief generation failed: {str(e)[:80]}")
            raise RuntimeError(f"Failed to generate content brief: {e}")


@dataclass
class ContentPublisher:
    llm: object
    logger: Callable[[str], None]

    def publish(self, topic: str, channel: str, audience: str, context: dict, brief: str) -> str:
        """Generate final content with error handling."""
        try:
            if not topic or not topic.strip():
                raise ValueError("Topic cannot be empty")
            if not channel:
                raise ValueError("Channel cannot be empty")
            if not context:
                raise ValueError("Context cannot be empty")
            
            ctx = build_prompt_context(topic, channel, audience, context)
            system_prompt, user_prompt = get_prompt_for_channel(ctx, brief=brief)
            
            if not user_prompt or not system_prompt:
                raise RuntimeError(f"Failed to build prompts for channel: {channel}")

            content_response = self.llm.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.75,
                max_tokens=800,
            )
            
            if not content_response or not content_response.content.strip():
                raise RuntimeError(f"LLM failed to generate {channel} content")

            self.logger(f"  OK Content generated ({content_response.total_tokens} tokens used)")
            return content_response.content
        except Exception as e:
            self.logger(f"  ✗ Content generation failed: {str(e)[:80]}")
            raise RuntimeError(f"Failed to generate content: {e}")


@dataclass
class OutputManager:
    output_dir: Path = OUTPUT_DIR

    def save_content(self, content: str, topic: str, channel: str, edited: bool = False) -> Path:
        """Save content with error handling."""
        try:
            if not content or not content.strip():
                raise ValueError("Content cannot be empty")
            if not topic or not topic.strip():
                raise ValueError("Topic cannot be empty")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_edited" if edited else ""
            filename = self.output_dir / f"{timestamp}_{channel}_{_slugify(topic)}{suffix}.txt"

            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(filename, "w", encoding="utf-8") as handle:
                handle.write(f"Topic: {topic}\n")
                handle.write(f"Channel: {channel}\n")
                handle.write(f"Generated: {datetime.now().isoformat()}\n")
                handle.write(f"Edited: {edited}\n")
                handle.write("\n" + "=" * 60 + "\n\n")
                handle.write(content)

            return filename
        except IOError as e:
            raise RuntimeError(f"Failed to save content to file: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while saving: {e}")

    def save_comparison(self, comparison: dict, topic: str) -> Path:
        """Save comparison result with error handling."""
        try:
            if not comparison or not isinstance(comparison, dict):
                raise ValueError("Invalid comparison data")
            if not topic or not topic.strip():
                raise ValueError("Topic cannot be empty")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"{timestamp}_uniqueness_comparison_{_slugify(topic, 30)}.json"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, "w", encoding="utf-8") as handle:
                json.dump(comparison, handle, indent=2)
            
            return filepath
        except IOError as e:
            raise RuntimeError(f"Failed to save comparison file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error saving comparison: {e}")


@dataclass
class ContentReviewer:
    output_manager: OutputManager
    input_fn: Callable[[str], str] = input
    print_fn: Callable[..., None] = print

    def _read_multiline(self) -> str:
        """Read user-edited text until two consecutive blank lines."""
        lines: list[str] = []
        blank_run = 0

        while True:
            try:
                line = self.input_fn("")
            except EOFError:
                break

            if line == "":
                blank_run += 1
            else:
                blank_run = 0

            lines.append(line)

            if blank_run >= 2:
                break

        while lines and lines[-1] == "":
            lines.pop()

        return "\n".join(lines)

    def review(self, content: str, topic: str, channel: str) -> str:
        self.print_fn("\n" + "=" * 60)
        self.print_fn(f"GENERATED {channel.upper()} CONTENT")
        self.print_fn("=" * 60)
        self.print_fn(content)
        self.print_fn("=" * 60)

        while True:
            try:
                choice = self.input_fn("\nApprove? [y = save / e = edit / r = regenerate / q = quit]: ").strip().lower()
            except EOFError:
                return ""

            if choice == "y":
                self.output_manager.save_content(content, topic, channel)
                self.print_fn("  OK Saved to outputs/")
                return content

            if choice == "e":
                self.print_fn("Enter your revised version (press Enter twice when done):")
                revised = self._read_multiline().strip()
                if not revised:
                    self.print_fn("  No changes entered.")
                    return content
                self.output_manager.save_content(revised, topic, channel, edited=True)
                self.print_fn("  OK Edited version saved.")
                return revised

            if choice == "r":
                self.print_fn("  Returning 'regenerate' signal...")
                return "__REGENERATE__"

            if choice == "q":
                self.print_fn("  Exiting without saving.")
                return ""

            self.print_fn("  Please enter y, e, r or q.")


def build_uniqueness_context(topic: str, channel: str, context: dict) -> PromptContext:
    """Create the context used for comparison runs."""
    return build_prompt_context(topic, channel, "fitness_enthusiast", context)


def run_uniqueness(topic: str, channel: str, context: dict, llm: object) -> dict:
    ctx = build_uniqueness_context(topic, channel, context)
    _, fitbyte_prompt = get_prompt_for_channel(ctx)

    return run_uniqueness_comparison(
        topic=topic,
        channel=channel,
        fitbyte_prompt=fitbyte_prompt,
        fitbyte_system=build_writer_system_prompt(ctx),
        llm=llm,
    )
