"""
knowledge_base.py
-----------------
Manages loading and accessing FitByte's two knowledge bases:

  PRIMARY  - brand guidelines, product specs, past content examples
  SECONDARY - market trends, competitor analysis

Provides clean accessor methods so the pipeline always gets
the right context without worrying about file paths.
"""

import re
from pathlib import Path

from document_processor import load_and_clean, load_file, truncate, extract_section, strip_markdown


KB_ROOT = Path(__file__).parent.parent / "knowledge_base"


class KnowledgeBase:
    """
    Central access point for all FitByte knowledge base content.

    Usage:
        kb = KnowledgeBase()
        brand_voice = kb.brand_voice()
        specs_summary = kb.product_specs(model="Pro")
        examples = kb.content_examples(n=3)
        market = kb.market_context()
    """

    def __init__(self, kb_root: Path = KB_ROOT):
        self.primary = kb_root / "primary"
        self.secondary = kb_root / "secondary"
        self._cache: dict[str, str] = {}
        self._raw_cache: dict[str, str] = {}

    def _load(self, path: Path) -> str:
        key = str(path)
        if key not in self._cache:
            self._cache[key] = load_and_clean(path)
        return self._cache[key]

    def _load_raw(self, path: Path) -> str:
        """Load raw markdown once and reuse it across section extractors."""
        key = str(path)
        if key not in self._raw_cache:
            self._raw_cache[key] = load_file(path)
        return self._raw_cache[key]

    @staticmethod
    def _bundle_sections(raw: str, sections: list[str], fallback: str = "") -> str:
        """Return a compact block of named sections from a markdown document."""
        parts = []
        for section in sections:
            content = extract_section(raw, section)
            if content:
                parts.append(f"[{section}]\n{content}")
        return "\n\n".join(parts) if parts else fallback

    @staticmethod
    def _select_diverse_posts(posts: list[str], n: int) -> list[str]:
        """Pick a small, varied sample of posts for prompt injection."""
        if n <= 0:
            return []
        if n >= len(posts):
            return posts
        if n == 1:
            return [posts[0]]
        if n == 2:
            return [posts[0], posts[-1]]

        selected = []
        for idx in [0, len(posts) // 2, len(posts) - 1]:
            post = posts[idx]
            if post not in selected:
                selected.append(post)
            if len(selected) == n:
                break
        return selected

    # ------------------------------------------------------------------ #
    # PRIMARY KNOWLEDGE BASE
    # ------------------------------------------------------------------ #

    def brand_voice(self) -> str:
        """
        Returns the brand voice and writing rules section -
        the most important context for tone and style.
        """
        raw = self._load_raw(self.primary / "fitbyte_brand_guidelines.md")
        sections = ["Our Voice", "Writing Rules", "Words We Use", "Tone by Channel", "What We Never Do"]
        return self._bundle_sections(raw, sections, fallback=self._load(self.primary / "fitbyte_brand_guidelines.md"))

    def brand_identity(self) -> str:
        """Returns the Who We Are + one-liner + audience descriptions."""
        raw = self._load_raw(self.primary / "fitbyte_brand_guidelines.md")
        sections = ["Who We Are", "Our Audience", "How to Talk About FitByte"]
        return self._bundle_sections(raw, sections)

    def product_specs(self, model: str = "all") -> str:
        """
        Returns product spec context.
        model: 'Core', 'Pro', 'Ultra', or 'all'
        """
        full = self._load(self.primary / "fitbyte_product_specs.md")
        if model == "all":
            return truncate(full, 2500)

        # Filter to sections mentioning the requested model
        lines = full.split("\n")
        relevant = [l for l in lines if model in l or not l.startswith("|")]
        return truncate("\n".join(relevant), 2000)

    def content_examples(self, n: int = 3) -> str:
        """
        Returns n example blog posts as style reference.
        Selects a diverse sample: recovery, sleep, and a seasonal/motivational post.
        """
        raw = self._load_raw(self.primary / "past_content" / "fitbyte_content_examples.md")

        # Extract individual posts by splitting on ## Post
        posts = re.split(r"(?=^## Post)", raw, flags=re.MULTILINE)
        posts = [p.strip() for p in posts if p.strip() and "## Post" in p]

        # Select a diverse sample
        selected = self._select_diverse_posts(posts, n)

        # Clean markdown for LLM injection
        cleaned = [strip_markdown(p) for p in selected]
        return "\n\n---\n\n".join(cleaned)

    def writing_rules(self) -> str:
        """Returns a compact cheat sheet of FitByte's do/don't writing rules."""
        raw = self._load_raw(self.primary / "fitbyte_brand_guidelines.md")
        rules = extract_section(raw, "Brand Voice Cheat Sheet")
        words = extract_section(raw, "Words We Use")
        parts = []
        if rules:
            parts.append(f"[Writing Rules Cheat Sheet]\n{rules}")
        if words:
            parts.append(f"[Vocabulary]\n{words}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # SECONDARY KNOWLEDGE BASE
    # ------------------------------------------------------------------ #

    def market_context(self, topic: str = "") -> str:
        """
        Returns relevant market trends and consumer insights.
        Optionally filter by topic keyword.
        """
        full = self._load(self.secondary / "market_trends.md")
        if not topic:
            return truncate(full, 2000)

        # Return paragraphs mentioning the topic
        paragraphs = full.split("\n\n")
        relevant = [p for p in paragraphs if topic.lower() in p.lower()]
        if relevant:
            return truncate("\n\n".join(relevant), 1500)
        return truncate(full, 1500)

    def competitor_context(self) -> str:
        """Returns key differentiators vs competitors - for positioning content."""
        return self._load(self.secondary / "competitor_analysis.md")

    def audience_pain_points(self) -> str:
        """Returns audience pain points from market research."""
        raw = self._load_raw(self.secondary / "market_trends.md")
        pain = extract_section(raw, "Audience Pain Points (Research Findings)")
        trends = extract_section(raw, "Key Consumer Trends")
        parts = []
        if trends:
            parts.append(f"[Consumer Trends]\n{trends}")
        if pain:
            parts.append(f"[Audience Pain Points]\n{pain}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # COMBINED CONTEXT
    # ------------------------------------------------------------------ #

    def full_context_for_generation(self, topic: str = "", audience: str = "") -> dict:
        """
        Returns a structured dict with all context needed for content generation.
        Used by the pipeline to build the final prompt.
        """
        return {
            "brand_voice": self.brand_voice(),
            "brand_identity": self.brand_identity(),
            "writing_rules": self.writing_rules(),
            "product_specs": self.product_specs(),
            "content_examples": self.content_examples(n=3),
            "market_context": self.market_context(topic=topic),
            "differentiators": self.competitor_context(),
            "audience_insights": self.audience_pain_points(),
        }


if __name__ == "__main__":
    kb = KnowledgeBase()
    print("=== Brand Voice (first 400 chars) ===")
    print(kb.brand_voice()[:400])
    print("\n=== Content Examples (first 300 chars) ===")
    print(kb.content_examples(n=2)[:300])
    print("\n=== Market Context (first 300 chars) ===")
    print(kb.market_context()[:300])
