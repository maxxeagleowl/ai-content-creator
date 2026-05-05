"""
document_processor.py
---------------------
Ingests and parses markdown files from the FitByte knowledge bases.
Extracts sections, strips markdown syntax, and returns clean text
ready to be injected into LLM prompts.
"""

import re
from pathlib import Path
from typing import Optional


def strip_markdown(text: str) -> str:
    """Remove markdown syntax and return plain readable text."""
    # Remove code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Remove inline code
    text = re.sub(r"`[^`]+`", lambda m: m.group()[1:-1], text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove image syntax
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove link syntax but keep text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Remove table separators
    text = re.sub(r"^\|[-| :]+\|$", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Remove heading markers but keep text
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Clean up extra blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_section(content: str, section_heading: str) -> Optional[str]:
    """
    Extract a specific section from markdown content by heading.

    Args:
        content: Full markdown file content
        section_heading: The heading to search for (case-insensitive)

    Returns:
        Section content as plain text, or None if not found
    """
    pattern = re.compile(
        rf"^#{{{1,6}}}\s+{re.escape(section_heading)}\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(content)
    if not match:
        return None

    heading_level = len(re.match(r"^(#{1,6})\s+", match.group(0)).group(1))
    start = match.end()

    # End the section at the next heading with the same or higher priority.
    # Lower heading levels use fewer # characters, so we only stop at headings
    # with a level less than or equal to the current one.
    end = len(content)
    for line_match in re.finditer(r"^(#{1,6})\s+.*$", content[start:], re.MULTILINE):
        next_level = len(line_match.group(1))
        if next_level <= heading_level:
            end = start + line_match.start()
            break

    return strip_markdown(content[start:end]).strip()


def load_file(path: str | Path) -> str:
    """Load a markdown file and return its raw content."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_and_clean(path: str | Path) -> str:
    """Load a markdown file and return stripped plain text."""
    return strip_markdown(load_file(path))


def load_all_from_directory(directory: str | Path) -> dict[str, str]:
    """
    Load all .md files from a directory.

    Returns:
        Dict mapping filename stem -> plain text content
    """
    directory = Path(directory)
    if not directory.exists():
        return {}

    result = {}
    for md_file in sorted(directory.rglob("*.md")):
        key = md_file.stem
        result[key] = load_and_clean(md_file)
    return result


def truncate(text: str, max_chars: int = 3000) -> str:
    """Truncate text to max_chars, breaking at a sentence boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(". ")
    if last_period > max_chars * 0.7:
        return truncated[: last_period + 1]
    return truncated + "..."


if __name__ == "__main__":
    # Quick smoke test
    base = Path(__file__).parent.parent / "knowledge_base"
    brand = load_and_clean(base / "primary" / "fitbyte_brand_guidelines.md")
    print(f"Brand guidelines loaded: {len(brand)} chars")
    print(brand[:300])
