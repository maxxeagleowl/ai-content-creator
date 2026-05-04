"""
prompt_templates.py
-------------------
Advanced prompt engineering templates for FitByte content generation.

Each template injects brand voice, product specifics, style examples,
and market context — ensuring outputs are brand-aligned and distinct
from generic AI-generated content.

Techniques used (from Module 2 lessons):
  - Role prompting (third-person persona)
  - Few-shot examples from the actual knowledge base
  - Chain-of-thought for brief generation
  - Contextual placement (unique brand POV injection)
  - Style variation (different tones per channel/audience)
  - XML-structured output for reliable parsing
"""

from dataclasses import dataclass
from typing import Literal

Channel = Literal["blog", "instagram", "linkedin", "email_subject"]
Audience = Literal["performance_athlete", "fitness_enthusiast", "health_professional", "upgrader", "general"]


@dataclass
class PromptContext:
    """All the context needed to build a generation prompt."""
    topic: str
    channel: Channel = "blog"
    audience: Audience = "general"
    brand_voice: str = ""
    writing_rules: str = ""
    content_examples: str = ""
    product_specs: str = ""
    market_context: str = ""
    differentiators: str = ""
    extra_instructions: str = ""


# ------------------------------------------------------------------ #
# SYSTEM PROMPTS (role definition)
# ------------------------------------------------------------------ #

SYSTEM_PROMPT_WRITER = """The assistant is an expert content writer for FitByte, a precision fitness watch brand. \
They have 10 years of experience writing health-tech content that is human, grounded and credible — never generic. \
They have deeply internalized FitByte's brand voice: clear not clinical, motivating not preachy, \
precise not verbose, confident not arrogant, human not robotic. \
They never use passive voice, serial commas, fear-based messaging or unqualified superlatives. \
Every word earns its place."""

SYSTEM_PROMPT_ANALYST = """The assistant is a senior content strategist for FitByte. \
They analyze topics through the lens of FitByte's audience, brand values and market positioning \
to produce sharp, actionable content briefs. \
They always connect product features to user benefits and know how to position FitByte \
credibly against the market without naming competitors."""


# ------------------------------------------------------------------ #
# TEMPLATE: CONTENT BRIEF (Step 3 of pipeline)
# ------------------------------------------------------------------ #

def build_brief_prompt(ctx: PromptContext) -> str:
    """
    Chain-of-thought prompt to generate a structured content brief.
    Forces the model to think about angle, hook, and differentiator
    before generating the actual post.
    """
    return f"""You are creating a content brief for a FitByte blog post about: "{ctx.topic}"

TARGET AUDIENCE: {ctx.audience.replace("_", " ").title()}
CHANNEL: {ctx.channel}

BRAND CONTEXT:
{ctx.brand_voice[:1200]}

MARKET CONTEXT:
{ctx.market_context[:800]}

PRODUCT DIFFERENTIATORS:
{ctx.differentiators[:600]}

Think step by step:

1. ANGLE: What specific angle makes this topic interesting for {ctx.audience.replace("_", " ")} readers? 
   What's the unexpected or counter-intuitive take?

2. HOOK: What opening line would stop someone mid-scroll? 
   (FitByte style: short, human, slightly provocative)

3. CORE INSIGHT: What is the one thing the reader should understand after reading this? 
   Which FitByte feature or data point supports this concretely?

4. MARKET CONTEXT: What consumer trend or pain point does this address?

5. CALL TO ACTION: What low-pressure action should the reader take? 
   (FitByte never hard-sells — end with something actionable or a quiet punchline)

Return your brief in this format:
ANGLE: [one sentence]
HOOK: [opening line]
CORE INSIGHT: [2-3 sentences]
MARKET RELEVANCE: [one sentence]
CTA: [one sentence]
WORD COUNT TARGET: [number, 150-250 for blog]
KEY PRODUCT REFERENCE: [which FitByte feature/metric to mention]"""


# ------------------------------------------------------------------ #
# TEMPLATE: BLOG POST (main generation)
# ------------------------------------------------------------------ #

def build_blog_post_prompt(ctx: PromptContext, brief: str = "") -> str:
    """
    Few-shot + contextual placement prompt for blog post generation.
    Uses real FitByte content examples to anchor style.
    """
    brief_section = f"""
CONTENT BRIEF TO FOLLOW:
{brief}
""" if brief else ""

    return f"""Write a FitByte blog post about: "{ctx.topic}"

{brief_section}

BRAND VOICE & WRITING RULES (follow these exactly):
{ctx.brand_voice[:1500]}

VOCABULARY RULES:
{ctx.writing_rules[:500]}

PRODUCT FACTS YOU CAN USE (only mention what's relevant — don't dump specs):
{ctx.product_specs[:1000]}

STYLE REFERENCE — these are real FitByte posts. Match this voice exactly:
{ctx.content_examples[:2000]}

---

WRITING INSTRUCTIONS:
- Length: 150-250 words
- Format: Heading (Title Case, punchy) + 3-4 short paragraphs
- Voice: Second person ("you"), conversational, warm but not fluffy
- No passive voice, no serial comma, no hype words without data
- Lead with a human observation or relatable scenario — NOT a product feature
- Weave in exactly ONE FitByte feature/metric naturally (don't start or end with it)
- End with something actionable or a quiet punchline — never a hard sell
- Every sentence must earn its place — cut anything that doesn't add meaning

Write the post now. Output only the post itself — no preamble, no notes."""


# ------------------------------------------------------------------ #
# TEMPLATE: INSTAGRAM CAPTION
# ------------------------------------------------------------------ #

def build_instagram_prompt(ctx: PromptContext) -> str:
    return f"""Write an Instagram caption for FitByte about: "{ctx.topic}"

BRAND VOICE:
{ctx.brand_voice[:800]}

STYLE RULES FOR INSTAGRAM:
- 1-2 lines maximum
- Lead with the hook — no warm-up
- Punchy, visual-first, slightly provocative
- No emojis in the main line (FitByte is understated)
- Optional: 1 relevant hashtag at the end
- DO NOT start with "Did you know" or "Here's why"

EXAMPLES OF GREAT FITBYTE CAPTIONS (adapt this energy):
- "Rest days aren't lazy. They're where fitness actually happens."
- "8 hours in bed isn't the same as 8 hours of sleep. FitByte shows you the difference."
- "Your body knew it was getting sick before your brain did."

Write exactly 1-2 lines. Output only the caption."""


# ------------------------------------------------------------------ #
# TEMPLATE: LINKEDIN POST
# ------------------------------------------------------------------ #

def build_linkedin_prompt(ctx: PromptContext) -> str:
    return f"""Write a LinkedIn post for FitByte about: "{ctx.topic}"

BRAND VOICE:
{ctx.brand_voice[:800]}

MARKET CONTEXT:
{ctx.market_context[:600]}

LINKEDIN TONE RULES:
- Credible, data-led, professional but human
- 100-150 words
- Health-tech angle with workplace wellness crossover
- Lead with a concrete insight or data point — not a question
- One FitByte capability mentioned naturally
- End with a thought-provoking statement (no CTA)
- No serial comma, no passive voice

Output only the LinkedIn post."""


# ------------------------------------------------------------------ #
# TEMPLATE: EMAIL SUBJECT LINES
# ------------------------------------------------------------------ #

def build_email_subjects_prompt(ctx: PromptContext) -> str:
    return f"""Generate 5 email subject line options for a FitByte campaign about: "{ctx.topic}"

BRAND RULES:
- Under 45 characters each
- Lead with numbers when possible (e.g. "7 days," "3 reasons")
- No ALL CAPS, no excessive punctuation
- Direct, benefit-led, slightly curious
- No spam trigger words

Return exactly 5 options, numbered, no other text."""


# ------------------------------------------------------------------ #
# TEMPLATE: UNIQUENESS COMPARISON
# ------------------------------------------------------------------ #

GENERIC_PROMPT = "Write a short blog post about {topic} for a fitness watch brand."

def build_generic_prompt(topic: str) -> str:
    """Returns the generic baseline prompt for uniqueness comparison."""
    return GENERIC_PROMPT.format(topic=topic)


# ------------------------------------------------------------------ #
# CHANNEL ROUTER
# ------------------------------------------------------------------ #

def get_prompt_for_channel(ctx: PromptContext, brief: str = "") -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for the given channel.
    """
    if ctx.channel == "blog":
        return SYSTEM_PROMPT_WRITER, build_blog_post_prompt(ctx, brief)
    elif ctx.channel == "instagram":
        return SYSTEM_PROMPT_WRITER, build_instagram_prompt(ctx)
    elif ctx.channel == "linkedin":
        return SYSTEM_PROMPT_WRITER, build_linkedin_prompt(ctx)
    elif ctx.channel == "email_subject":
        return SYSTEM_PROMPT_WRITER, build_email_subjects_prompt(ctx)
    else:
        return SYSTEM_PROMPT_WRITER, build_blog_post_prompt(ctx, brief)


if __name__ == "__main__":
    ctx = PromptContext(
        topic="why your resting heart rate matters more than step count",
        channel="blog",
        audience="fitness_enthusiast"
    )
    print(build_brief_prompt(ctx)[:500])
