"""
content_pipeline.py
-------------------
Orchestrates the FitByte content creation pipeline:

  1. DOCUMENT - Load and structure context from both knowledge bases
  2. MONITOR  - Analyze topic against market trends and brand fit
  3. BRIEF    - Generate a structured content brief
  4. PUBLISH  - Generate final content for the target channel
  5. ITERATE  - Human review, feedback loop, and output saving
"""

from __future__ import annotations

import time

from knowledge_base import KnowledgeBase
from llm_integration import get_llm_client
from pipeline_components import (
    ContentBriefGenerator,
    ContentDocumenter,
    ContentMonitor,
    ContentPublisher,
    ContentReviewer,
    OutputManager,
    run_uniqueness,
)


class ContentPipeline:
    """
    Full content creation pipeline for FitByte.

    Steps:
      document -> monitor -> brief -> publish -> iterate
    """

    def __init__(self, provider: str = "auto", verbose: bool = True, batch_delay_seconds: float = 1.0):
        self.kb = KnowledgeBase()
        self.llm = get_llm_client(provider)
        self.verbose = verbose
        self.batch_delay_seconds = batch_delay_seconds

        self.outputs = OutputManager()
        self.documenter = ContentDocumenter(self.kb)
        self.monitorer = ContentMonitor(self.llm, self._log)
        self.brief_generator = ContentBriefGenerator(self.llm, self._log)
        self.publisher = ContentPublisher(self.llm, self._log)
        self.reviewer = ContentReviewer(self.outputs)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _header(self, topic: str, channel: str, audience: str):
        self._log(f"\n{'=' * 60}")
        self._log("FitByte Content Pipeline")
        self._log(f"Topic: {topic}")
        self._log(f"Channel: {channel} | Audience: {audience}")
        self._log(f"{'=' * 60}")

    # ------------------------------------------------------------------ #
    # STEP 1: DOCUMENT
    # ------------------------------------------------------------------ #

    def document(self, topic: str, audience: str) -> dict:
        """Load and structure all relevant context from knowledge bases."""
        self._log("\n[1/5] DOCUMENT - Loading knowledge bases...")
        context = self.documenter.document(topic, audience)
        self._log(f"  OK Brand voice: {len(context['brand_voice'])} chars")
        self._log(f"  OK Product specs: {len(context['product_specs'])} chars")
        self._log(f"  OK Content examples: {len(context['content_examples'])} chars")
        self._log(f"  OK Market context: {len(context['market_context'])} chars")
        return context

    # ------------------------------------------------------------------ #
    # STEP 2: MONITOR
    # ------------------------------------------------------------------ #

    def monitor(self, topic: str, context: dict) -> dict:
        """Analyze the topic for brand fit, market relevance, and content angle."""
        self._log("\n[2/5] MONITOR - Analyzing topic...")
        return self.monitorer.analyze(topic, context)

    # ------------------------------------------------------------------ #
    # STEP 3: BRIEF
    # ------------------------------------------------------------------ #

    def brief(self, topic: str, channel: str, audience: str, context: dict) -> str:
        """Generate a structured content brief."""
        self._log("\n[3/5] BRIEF - Generating content brief...")
        brief = self.brief_generator.generate(topic, channel, audience, context)
        if self.verbose:
            print(f"\n  --- BRIEF ---\n{brief}\n  ---")
        return brief

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
        self._log(f"\n[4/5] PUBLISH - Generating {channel} content...")
        return self.publisher.publish(topic, channel, audience, context, brief)

    # ------------------------------------------------------------------ #
    # STEP 5: ITERATE
    # ------------------------------------------------------------------ #

    def iterate(self, content: str, topic: str, channel: str) -> str:
        """Human-in-the-loop review step."""
        self._log("\n[5/5] ITERATE - Human review")
        return self.reviewer.review(content, topic, channel)

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
        """Run the full pipeline end-to-end."""
        self._header(topic, channel, audience)

        context = self.document(topic, audience)
        monitor_report = self.monitor(topic, context)
        content_brief = self.brief(topic, channel, audience, context)
        content = self.publish(topic, channel, audience, context, content_brief)

        if auto_approve:
            self.outputs.save_content(content, topic, channel)
            final_content = content
        else:
            current_content = content
            while True:
                final_content = self.iterate(current_content, topic, channel)
                if final_content != "__REGENERATE__":
                    break

                self._log("  Regenerating...")
                current_content = self.publish(topic, channel, audience, context, content_brief)

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
        """Run multiple content requests without human review."""
        results = []
        for i, req in enumerate(requests, 1):
            print(f"\n[Batch {i}/{len(requests)}] {req.get('topic', '')[:60]}")
            result = self.run(**req, auto_approve=True)
            results.append(result)
            if self.batch_delay_seconds > 0:
                time.sleep(self.batch_delay_seconds)
        return results

    # ------------------------------------------------------------------ #
    # UNIQUENESS COMPARISON
    # ------------------------------------------------------------------ #

    def compare_uniqueness(self, topic: str, channel: str = "blog") -> dict:
        """Run a side-by-side comparison: FitByte branded vs generic prompt."""
        self._log(f"\n{'=' * 60}")
        self._log("UNIQUENESS COMPARISON")
        self._log(f"Topic: {topic}")
        self._log(f"{'=' * 60}")

        context = self.document(topic, "fitness_enthusiast")
        comparison = run_uniqueness(topic, channel, context, self.llm)

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

        filepath = self.outputs.save_comparison(comparison, topic)
        self._log(f"\n  OK Comparison saved to {filepath.name}")

        return comparison
