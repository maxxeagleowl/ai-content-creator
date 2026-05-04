"""
content_pipeline.py
-------------------
Orchestrates the FitByte content creation pipeline:

  1. DOCUMENT  — Load and structure context from both knowledge bases
  2. MONITOR   — Analyze topic against market trends and brand fit
  3. BRIEF     — Generate a structured content brief
  4. PUBLISH   — Generate final content for the target channel
  5. ITERATE   — Human review, feedback loop, and output saving

Usage:
    from content_pipeline import ContentPipeline
    pipeline = ContentPipeline()
    result = pipeline.run(
        topic="why rest days matter more than you think",
        channel="blog",
        audience="fitness_enthusiast"
    )
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from knowledge_base import KnowledgeBase
from llm_integration import get_llm_client, run_uniqueness_comparison
from prompt_templates import (
    PromptContext,
    build_brief_prompt,
    get_prompt_for_channel,
    SYSTEM_PROMPT_ANALYST,
)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


class ContentPipeline:
    """
    Full content creation pipeline for FitByte.

    Steps:
      document → monitor → brief → publish → iterate
    """

    def __init__(self, provider: str = "auto", verbose: bool = True):
        self.kb = KnowledgeBase()
        self.llm = get_llm_client(provider)
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------ #
    # STEP 1: DOCUMENT
    # ------------------------------------------------------------------ #

    def document(self, topic: str, audience: str) -> dict:
        """Load and structure all relevant context from knowledge bases."""
        self._log("\n[1/5] DOCUMENT — Loading knowledge bases...")
        context = self.kb.full_context_for_generation(topic=topic, audience=audience)
        self._log(f"  ✓ Brand voice: {len(context['brand_voice'])} chars")
        self._log(f"  ✓ Product specs: {len(context['product_specs'])} chars")
        self._log(f"  ✓ Content examples: {len(context['content_examples'])} chars")
        self._log(f"  ✓ Market context: {len(context['market_context'])} chars")
        return context

    # ------------------------------------------------------------------ #
    # STEP 2: MONITOR
    # ------------------------------------------------------------------ #

    def monitor(self, topic: str, context: dict) -> dict:
        """
        Analyze the topic for brand fit, market relevance, and content angle.
        Returns a monitoring report.
        """
        self._log("\n[2/5] MONITOR — Analyzing topic...")

        analysis_prompt = f"""You are a content strategist for FitByte, a precision fitness watch brand.

Analyze this content topic: "{topic}"

Use this market context:
{context['market_context'][:800]}

FitByte brand identity:
{context['brand_identity'][:600]}

Answer briefly:
1. BRAND FIT (1-10): How well does this topic align with FitByte's voice and audience?
2. MARKET RELEVANCE: Which consumer trend does this address?
3. BEST ANGLE: What's the most distinctive angle for FitByte specifically?
4. RISK: Any brand voice pitfalls to avoid for this topic?

Return as JSON with keys: brand_fit_score, market_relevance, best_angle, risk"""

        response = self.llm.generate(
            user_prompt=analysis_prompt,
            system_prompt=SYSTEM_PROMPT_ANALYST,
            temperature=0,
            max_tokens=400,
        )

        try:
            # Strip any markdown code fences if present
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            report = json.loads(raw.strip())
        except json.JSONDecodeError:
            report = {"raw_analysis": response.content}

        self._log(f"  ✓ Brand fit score: {report.get('brand_fit_score', 'N/A')}/10")
        self._log(f"  ✓ Best angle: {report.get('best_angle', 'N/A')[:80]}")
        return report

    # ------------------------------------------------------------------ #
    # STEP 3: BRIEF
    # ------------------------------------------------------------------ #

    def brief(self, topic: str, channel: str, audience: str, context: dict) -> str:
        """Generate a structured content brief using chain-of-thought prompting."""
        self._log("\n[3/5] BRIEF — Generating content brief...")

        ctx = PromptContext(
            topic=topic,
            channel=channel,
            audience=audience,
            brand_voice=context["brand_voice"],
            writing_rules=context["writing_rules"],
            market_context=context["market_context"],
            differentiators=context["differentiators"],
        )

        brief_response = self.llm.generate(
            user_prompt=build_brief_prompt(ctx),
            system_prompt=SYSTEM_PROMPT_ANALYST,
            temperature=0.5,
            max_tokens=600,
        )

        self._log(f"  ✓ Brief generated ({brief_response.total_tokens} tokens used)")
        if self.verbose:
            print(f"\n  --- BRIEF ---\n{brief_response.content}\n  ---")

        return brief_response.content

    # ------------------------------------------------------------------ #
    # STEP 4: PUBLISH
    # ------------------------------------------------------------------ #

    def publish(
        self,
        topic: str,
        channel: str,
        audience: str,
        context: dict,
        brief: str,
    ) -> str:
        """Generate final content using the brief and full brand context."""
        self._log(f"\n[4/5] PUBLISH — Generating {channel} content...")

        ctx = PromptContext(
            topic=topic,
            channel=channel,
            audience=audience,
            brand_voice=context["brand_voice"],
            writing_rules=context["writing_rules"],
            content_examples=context["content_examples"],
            product_specs=context["product_specs"],
            market_context=context["market_context"],
            differentiators=context["differentiators"],
        )

        system_prompt, user_prompt = get_prompt_for_channel(ctx, brief=brief)

        content_response = self.llm.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.75,
            max_tokens=800,
        )

        self._log(f"  ✓ Content generated ({content_response.total_tokens} tokens used)")
        return content_response.content

    # ------------------------------------------------------------------ #
    # STEP 5: ITERATE
    # ------------------------------------------------------------------ #

    def iterate(self, content: str, topic: str, channel: str) -> str:
        """
        Human-in-the-loop review step.
        Prints the content and prompts for approval, edit, or rejection.
        Returns the final approved content.
        """
        self._log("\n[5/5] ITERATE — Human review")
        print("\n" + "=" * 60)
        print(f"GENERATED {channel.upper()} CONTENT")
        print("=" * 60)
        print(content)
        print("=" * 60)

        while True:
            choice = input("\nApprove? [y = save / e = edit / r = regenerate / q = quit]: ").strip().lower()

            if choice == "y":
                self._save_output(content, topic, channel)
                print(f"  ✓ Saved to outputs/")
                return content

            elif choice == "e":
                print("Enter your revised version (press Enter twice when done):")
                lines = []
                while True:
                    line = input()
                    if line == "" and lines and lines[-1] == "":
                        break
                    lines.append(line)
                content = "\n".join(lines[:-1])  # Remove final blank line
                self._save_output(content, topic, channel, edited=True)
                print(f"  ✓ Edited version saved.")
                return content

            elif choice == "r":
                print("  Returning 'regenerate' signal...")
                return "__REGENERATE__"

            elif choice == "q":
                print("  Exiting without saving.")
                return ""

            else:
                print("  Please enter y, e, r or q.")

    # ------------------------------------------------------------------ #
    # SAVE OUTPUT
    # ------------------------------------------------------------------ #

    def _save_output(self, content: str, topic: str, channel: str, edited: bool = False):
        """Save generated content to the outputs directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = topic[:40].lower().replace(" ", "_").replace("/", "-")
        suffix = "_edited" if edited else ""
        filename = OUTPUT_DIR / f"{timestamp}_{channel}_{slug}{suffix}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Topic: {topic}\n")
            f.write(f"Channel: {channel}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Edited: {edited}\n")
            f.write("\n" + "=" * 60 + "\n\n")
            f.write(content)

        return filename

    # ------------------------------------------------------------------ #
    # MAIN RUN METHOD
    # ------------------------------------------------------------------ #

    def run(
        self,
        topic: str,
        channel: str = "blog",
        audience: str = "fitness_enthusiast",
        auto_approve: bool = False,
    ) -> dict:
        """
        Run the full pipeline end-to-end.

        Args:
            topic: Content topic or angle
            channel: 'blog', 'instagram', 'linkedin', 'email_subject'
            audience: 'performance_athlete', 'fitness_enthusiast',
                      'health_professional', 'upgrader', 'general'
            auto_approve: Skip human review (for batch mode)

        Returns:
            Dict with brief, content, monitor_report, and metadata
        """
        self._log(f"\n{'='*60}")
        self._log(f"FitByte Content Pipeline")
        self._log(f"Topic: {topic}")
        self._log(f"Channel: {channel} | Audience: {audience}")
        self._log(f"{'='*60}")

        # Steps 1-4
        context = self.document(topic, audience)
        monitor_report = self.monitor(topic, context)
        content_brief = self.brief(topic, channel, audience, context)
        content = self.publish(topic, channel, audience, context, content_brief)

        # Step 5 - iterate or auto-approve
        if auto_approve:
            self._save_output(content, topic, channel)
            final_content = content
        else:
            final_content = self.iterate(content, topic, channel)
            if final_content == "__REGENERATE__":
                self._log("  Regenerating...")
                content = self.publish(topic, channel, audience, context, content_brief)
                final_content = self.iterate(content, topic, channel)

        return {
            "topic": topic,
            "channel": channel,
            "audience": audience,
            "brief": content_brief,
            "content": final_content,
            "monitor_report": monitor_report,
        }

    # ------------------------------------------------------------------ #
    # BATCH MODE
    # ------------------------------------------------------------------ #

    def run_batch(self, requests: list[dict]) -> list[dict]:
        """
        Run multiple content requests without human review.

        Args:
            requests: List of dicts, each with keys: topic, channel, audience

        Returns:
            List of result dicts
        """
        results = []
        for i, req in enumerate(requests, 1):
            print(f"\n[Batch {i}/{len(requests)}] {req.get('topic', '')[:60]}")
            result = self.run(**req, auto_approve=True)
            results.append(result)
            import time
            time.sleep(1)  # Rate limit courtesy
        return results

    # ------------------------------------------------------------------ #
    # UNIQUENESS COMPARISON
    # ------------------------------------------------------------------ #

    def compare_uniqueness(self, topic: str, channel: str = "blog") -> dict:
        """
        Run a side-by-side comparison: FitByte branded vs generic prompt.
        Returns comparison dict and prints formatted results.
        """
        self._log(f"\n{'='*60}")
        self._log(f"UNIQUENESS COMPARISON")
        self._log(f"Topic: {topic}")
        self._log(f"{'='*60}")

        context = self.document(topic, "fitness_enthusiast")
        ctx = PromptContext(
            topic=topic,
            channel=channel,
            brand_voice=context["brand_voice"],
            writing_rules=context["writing_rules"],
            content_examples=context["content_examples"],
            product_specs=context["product_specs"],
            market_context=context["market_context"],
        )

        from prompt_templates import SYSTEM_PROMPT_WRITER
        _, fitbyte_prompt = get_prompt_for_channel(ctx)

        comparison = run_uniqueness_comparison(
            topic=topic,
            fitbyte_prompt=fitbyte_prompt,
            fitbyte_system=SYSTEM_PROMPT_WRITER,
            llm=self.llm,
        )

        # Print formatted comparison
        print("\n" + "=" * 60)
        print("FITBYTE BRANDED OUTPUT:")
        print("=" * 60)
        print(comparison["fitbyte_output"])
        print("\n" + "=" * 60)
        print("GENERIC OUTPUT (baseline):")
        print("=" * 60)
        print(comparison["generic_output"])
        print("\n" + "=" * 60)
        print("ANALYSIS:")
        print("=" * 60)
        print(comparison["analysis"])

        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = topic[:30].lower().replace(" ", "_")
        filepath = OUTPUT_DIR / f"{timestamp}_uniqueness_comparison_{slug}.json"
        with open(filepath, "w") as f:
            json.dump(comparison, f, indent=2)
        self._log(f"\n  ✓ Comparison saved to {filepath.name}")

        return comparison
