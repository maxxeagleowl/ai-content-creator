"""
prompt_templates.py

This module contains all prompt engineering logic for the AI Content Creator.

Purpose:
- Build structured prompts for different content types
- Inject knowledge base context into prompts
- Ensure outputs are brand-aligned and non-generic
- Support style variation across channels

Core idea:
We do NOT use static prompts.
We dynamically construct prompts using a PromptContext object.
"""

from dataclasses import dataclass
from typing import Literal


# ------------------------------------------------------------------
# TYPE DEFINITIONS
# ------------------------------------------------------------------

# Supported output channels
Channel = Literal["blog", "instagram", "linkedin", "email_subject"]

# Supported audience segments
Audience = Literal[
    "performance_athlete",
    "fitness_enthusiast",
    "health_professional",
    "upgrader",
    "general",
]


# ------------------------------------------------------------------
# CONTEXT OBJECT
# ------------------------------------------------------------------

NO_EM_DASH_RULE = (
    "Write like a real person. Avoid AI-sounding phrasing. "
    "Never use the em dash character — in any output; use a comma, colon, semicolon, or a plain hyphen instead."
)

NO_HYPHEN_RULE = (
    "Write in full sentences only. "
    "Do not use the hyphen character `-`."
)

def _append_output_rule(prompt: str) -> str:
    """Attach the shared output-style rule to a prompt."""
    return f"{prompt}\n\nOUTPUT RULE:\n{NO_EM_DASH_RULE}"

@dataclass
class PromptContext:
    """
    Central data container for all prompt inputs.

    This object is populated from the knowledge base and user input,
    and then passed into prompt templates.

    Design goal:
    - decouple prompt logic from data loading
    - keep templates clean and reusable
    """

    topic: str

    # brand-level context
    brand_name: str = "FitByte"
    industry: str = "fitness technology"

    # generation settings
    channel: Channel = "blog"
    audience: Audience = "general"

    # knowledge base inputs
    brand_identity: str = ""
    brand_voice: str = ""
    writing_rules: str = ""
    content_examples: str = ""
    product_specs: str = ""
    market_context: str = ""
    differentiators: str = ""
    audience_insights: str = ""

    # optional additional control
    extra_instructions: str = ""


# ------------------------------------------------------------------
# SYSTEM PROMPTS (ROLE DEFINITIONS)
# ------------------------------------------------------------------

def build_writer_system_prompt(ctx: PromptContext) -> str:
    """
    Defines the "writer persona" for content generation.

    This is used as the system message in the LLM call.
    It enforces tone, style, and quality expectations.
    """
    return f"""
The assistant is an expert content writer for {ctx.brand_name}, operating in the {ctx.industry} industry.

They produce content that is:
- human
- precise
- grounded in real context
- non-generic

Brand identity:
{ctx.brand_identity[:800]}

Audience insights:
{ctx.audience_insights[:800]}

They strictly follow the provided knowledge base context.
They never invent facts.
They avoid vague AI phrasing.
{NO_EM_DASH_RULE}
{NO_HYPHEN_RULE if ctx.channel in ('blog', 'linkedin') else ''}

Every sentence must add value.
""".strip()


def build_analyst_system_prompt(ctx: PromptContext) -> str:
    """
    Defines the "analyst persona" for brief generation.

    This is used in the first step of the pipeline
    to create structured content briefs.
    """
    return f"""
The assistant is a senior content strategist for {ctx.brand_name}.

They:
- analyze topics using brand and market context
- connect product features to user value
- produce structured, actionable content briefs
- focus on clarity and positioning
""".strip()


# ------------------------------------------------------------------
# STEP 1: CONTENT BRIEF GENERATION
# ------------------------------------------------------------------

def build_brief_prompt(ctx: PromptContext) -> str:
    """
    Builds a structured reasoning prompt to generate a content brief.

    This step forces the model to:
    - define angle
    - define hook
    - define core insight

    before generating final content.

    This improves output quality and uniqueness.
    """
    return _append_output_rule(f"""
Create a structured content brief for {ctx.brand_name}.

TOPIC:
{ctx.topic}

TARGET AUDIENCE:
{ctx.audience.replace("_", " ").title()}

CHANNEL:
{ctx.channel}

BRAND CONTEXT:
{ctx.brand_identity[:1200]}

AUDIENCE INSIGHTS:
{ctx.audience_insights[:1000]}

{ctx.brand_voice[:1500]}

MARKET CONTEXT:
{ctx.market_context[:1000]}

PRODUCT DIFFERENTIATORS:
{ctx.differentiators[:800]}

PRODUCT FACTS:
{ctx.product_specs[:800]}

TASK:
Analyze the topic and define a strong content direction.

Return this structure:

ANGLE:
HOOK:
CORE INSIGHT:
MARKET RELEVANCE:
PRODUCT REFERENCE:
STYLE DIRECTION:
CALL TO ACTION OR CLOSING:
""".strip())


# ------------------------------------------------------------------
# STEP 2: CONTENT GENERATION
# ------------------------------------------------------------------

def build_blog_post_prompt(ctx: PromptContext, brief: str = "") -> str:
    """
    Builds the final blog post generation prompt.

    Combines:
    - brief (optional)
    - brand context
    - product context
    - market context
    - style examples

    This is the core prompt for content creation.
    """

    # Optional brief injection
    brief_section = f"\nCONTENT BRIEF:\n{brief}\n" if brief else ""

    return _append_output_rule(f"""
Write a blog post for {ctx.brand_name} about:

{ctx.topic}

{brief_section}

BRAND VOICE:
{ctx.brand_identity[:1200]}

AUDIENCE INSIGHTS:
{ctx.audience_insights[:1000]}

{ctx.brand_voice[:1500]}

WRITING RULES:
{ctx.writing_rules[:800]}

PRODUCT FACTS:
{ctx.product_specs[:1200]}

MARKET CONTEXT:
{ctx.market_context[:1000]}

STYLE EXAMPLES:
{ctx.content_examples[:1800]}

INSTRUCTIONS:
- 150 to 250 words
- title plus short paragraphs
- lead with human insight
- include one product reference naturally
- avoid generic phrasing
- avoid unsupported claims

Output only the blog post.
{NO_HYPHEN_RULE}
""".strip())


def build_instagram_prompt(ctx: PromptContext) -> str:
    """
    Prompt for short-form, high-impact Instagram captions.
    Focus on brevity and strong hooks.
    """
    return _append_output_rule(f"""
Write an Instagram caption for {ctx.brand_name} about:

{ctx.topic}

RULES:
- start with one strong hookline
- after the hookline, write 2 short sentences about the user topic
- finish with 1 closing sentence that wraps the idea up
- no generic phrases
- minimal explanation
- use emojis 
- get to the point quickly
- use a very conversational, gen z tone

Output only the caption.
""".strip())


def build_linkedin_prompt(ctx: PromptContext) -> str:
    """
    Prompt for professional LinkedIn content.

    Emphasizes:
    - credibility
    - business relevance
    - structured thinking
    """
    return _append_output_rule(f"""
Write a LinkedIn post for {ctx.brand_name} about:

{ctx.topic}

BRAND IDENTITY:
{ctx.brand_identity[:900]}

AUDIENCE INSIGHTS:
{ctx.audience_insights[:800]}

RULES:
- 250 to 350 words
- start with a concrete insight
- craft a powerful hook that speaks to target audience pain points
- include one product capability
- no generic language
- have a numbered list of 3 key takeaways that are relevant to the audience
- end with a strong closing and a call to action (e.g. "What do you think?", "How are you approaching this?")
- include 3-5 relevant hashtags at the end

Output only the post.
{NO_HYPHEN_RULE}
""".strip())


def build_email_subjects_prompt(ctx: PromptContext) -> str:
    """
    Prompt for generating multiple email subject lines.
    """
    return _append_output_rule(f"""
Generate 5 email subject lines for {ctx.brand_name} about:

{ctx.topic}

RULES:
- under 15 characters
- direct and benefit-led
- no spam wording
- Start each with a verb (e.g. "Discover", "Unlock", "Boost")
- End with an emoji relevant to the topic (e.g. "🚀", "💪", "📈")
- Put Numbering on each subject line (e.g. "1. Discover...", "2. Unlock...")

Return exactly 5 lines.
""".strip())


# ------------------------------------------------------------------
# BASELINE PROMPT (FOR UNIQUENESS COMPARISON)
# ------------------------------------------------------------------

GENERIC_PROMPT = {
    "blog": "Write a short blog post about {topic} for a {industry} brand.",
    "instagram": "Write an Instagram caption about {topic} for a {industry} brand.",
    "linkedin": "Write a LinkedIn post about {topic} for a {industry} brand.",
    "email_subject": "Generate 5 email subject lines about {topic} for a {industry} brand.",
}


def build_generic_prompt(
    topic: str,
    channel: str = "blog",
    industry: str = "fitness technology",
) -> str:
    """
    Returns a generic prompt without context.

    Used to compare:
    - generic output vs system output
    """
    template = GENERIC_PROMPT.get(channel, GENERIC_PROMPT["blog"])
    return template.format(topic=topic, industry=industry)


# ------------------------------------------------------------------
# ROUTER
# ------------------------------------------------------------------

def get_prompt_for_channel(ctx: PromptContext, brief: str = "") -> tuple[str, str]:
    """
    Selects the correct prompt based on channel.

    Returns:
    (system_prompt, user_prompt)
    """

    system_prompt = build_writer_system_prompt(ctx)

    if ctx.channel == "blog":
        return system_prompt, build_blog_post_prompt(ctx, brief)

    if ctx.channel == "instagram":
        return system_prompt, build_instagram_prompt(ctx)

    if ctx.channel == "linkedin":
        return system_prompt, build_linkedin_prompt(ctx)

    if ctx.channel == "email_subject":
        return system_prompt, build_email_subjects_prompt(ctx)

    # fallback
    return system_prompt, build_blog_post_prompt(ctx, brief)


def get_brief_prompts(ctx: PromptContext) -> tuple[str, str]:
    """
    Returns prompts for the brief generation step.
    """
    return build_analyst_system_prompt(ctx), build_brief_prompt(ctx)
