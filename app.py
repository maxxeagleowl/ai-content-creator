"""
app.py
------
FitByte AI Content Creator — Gradio Interface

Workflow:
  1. User fills in topic, audience, channel, and optional custom instructions
  2. Click "Generate" → runs the full pipeline (Document → Monitor → Brief → Publish)
  3. Editable text area shows the generated blog post
  4. User can make manual edits directly in the text area
  5. "Refine with AI" lets the user type adjustment instructions → AI rewrites
  6. "Approve & Download" saves the final post as a .txt file

Run with:
    python app.py
"""

import sys
import os
import re
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from knowledge_base import KnowledgeBase
from llm_integration import get_llm_client
from prompt_templates import (
    PromptContext,
    build_analyst_system_prompt,
    build_brief_prompt,
    build_blog_post_prompt,
    build_writer_system_prompt,
)

# ── INIT ──────────────────────────────────────────────────────────────
kb = KnowledgeBase()
llm = get_llm_client(provider="auto")

# Cache for knowledge base contexts to avoid reloading
_kb_cache = {}

def get_cached_context(topic, audience):
    """Return cached KB context or generate and cache it."""
    cache_key = f"{topic}:{audience}"
    if cache_key not in _kb_cache:
        _kb_cache[cache_key] = kb.full_context_for_generation(topic=topic, audience=audience)
    return _kb_cache[cache_key]

AUDIENCES = {
    "Fitness Enthusiast": "fitness_enthusiast",
    "Performance Athlete": "performance_athlete",
    "Health-Conscious Professional": "health_professional",
    "Upgrader (switching from another watch)": "upgrader",
    "General": "general",
}

CHANNELS = {
    "Blog Post": "blog",
    "Instagram Caption": "instagram",
    "LinkedIn Post": "linkedin",
    "Email Subject Lines": "email_subject",
}

# ── GENERATION LOGIC ──────────────────────────────────────────────────

def generate_post(topic, audience_label, channel_label, custom_instructions, progress=gr.Progress()):
    """Generate content and show all UI elements."""
    
    if not topic.strip():
        gr.Warning("Please enter a topic before generating.")
        return "", gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(value="0 words")

    try:
        audience = AUDIENCES.get(audience_label, "fitness_enthusiast")
        channel = CHANNELS.get(channel_label, "blog")

        progress(0.20, desc="Loading knowledge bases...")
        context = get_cached_context(topic, audience)  # Use cached KB

        progress(0.50, desc="Generating content...")
        
        # Build context with brief generation skipped for speed
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
            extra_instructions=custom_instructions or "",
        )

        # OPTIMIZED: Skip brief generation and go straight to content
        # Use a simplified prompt that includes key brief elements inline
        blog_prompt = build_blog_post_prompt(ctx, brief="")  # No separate brief
        
        # Inject custom instructions into the generation prompt
        if custom_instructions.strip():
            blog_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{custom_instructions.strip()}"

        # Single LLM call with reduced max_tokens for faster response
        content_response = llm.generate(
            user_prompt=blog_prompt,
            system_prompt=build_writer_system_prompt(ctx),
            temperature=0.75,
            max_tokens=500,  # Reduced from 900 for faster generation
            timeout=45  # Add explicit timeout
        )

        if not content_response or not content_response.content.strip():
            gr.Error("Failed to generate content. Please try again.")
            return "", gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(value="0 words")

        progress(1.0, desc="Done!")

        content = content_response.content.strip()

        # Update word count and clear refine input
        word_count_text = word_count(content)
        
        return (
            content,
            gr.update(visible=True),   # show refinement row
            gr.update(visible=True),   # show approve row
            gr.update(value=""),      # clear refine input
            gr.update(value=word_count_text),  # update word count
        )
    
    except TimeoutError:
        gr.Error("Request timed out. Please try again with a shorter topic or simpler instructions.")
        return "", gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(value="0 words")
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR in generate_post: {error_msg}")
        gr.Error(f"Error generating content: {error_msg[:100]}")
        return "", gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(value="0 words")


def refine_post(current_content, refinement_instruction, topic, audience_label, channel_label):
    """Takes the current post + user instruction and asks the LLM to rewrite."""

    if not current_content.strip():
        gr.Warning("Generate a post first before refining.")
        return current_content

    if not refinement_instruction.strip():
        gr.Warning("Enter a refinement instruction.")
        return current_content

    try:
        audience = AUDIENCES.get(audience_label, "fitness_enthusiast")
        channel  = CHANNELS.get(channel_label, "blog")
        context  = get_cached_context(topic, audience)

        refine_prompt = f"""You are editing an existing FitByte blog post. Apply ONLY the instruction below — keep everything else the same.

CURRENT POST:
{current_content}

INSTRUCTION TO APPLY:
{refinement_instruction.strip()}

BRAND RULES (do not break these):
{context['writing_rules'][:400]}

Return ONLY the revised post — no preamble, no notes, no explanation."""

        response = llm.generate(
            user_prompt=refine_prompt,
            system_prompt=build_writer_system_prompt(
                PromptContext(topic=topic, channel=channel, audience=audience)
            ),
            temperature=0.65,
            max_tokens=600,
            timeout=45,
        )
        
        if not response or not response.content.strip():
            gr.Error("Failed to refine post. Please try again.")
            return current_content
        
        return response.content.strip()
    
    except TimeoutError:
        gr.Error("Refinement timed out. Please try a simpler instruction.")
        return current_content
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR in refine_post: {error_msg}")
        gr.Error(f"Error refining post: {error_msg[:100]}")
        return current_content


def approve_and_download(content, topic, channel_label):
    """Saves the approved post as a .txt file and returns the path for download."""

    if not content.strip():
        gr.Warning("Nothing to download — generate a post first.")
        return None

    try:
        channel = CHANNELS.get(channel_label, "blog")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "_", topic.lower().strip())[:40]
        filename = f"fitbyte_{channel}_{slug}_{timestamp}.txt"

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False,
            prefix=f"fitbyte_{channel}_"
        )
        tmp.write(f"FitByte Content — Approved\n")
        tmp.write(f"Topic:   {topic}\n")
        tmp.write(f"Channel: {channel_label}\n")
        tmp.write(f"Date:    {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        tmp.write("\n" + "=" * 60 + "\n\n")
        tmp.write(content)
        tmp.close()

        gr.Info(f"File saved successfully!")
        return tmp.name
    
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR in approve_and_download: {error_msg}")
        gr.Error(f"Error saving file: {error_msg[:100]}")
        return None


def word_count(text):
    if not text:
        return "0 words"
    count = len(text.split())
    color = "green" if 200 <= count <= 400 else ("orange" if count < 200 else "orange")
    return f"{count} words"


# ── THEME & CSS ───────────────────────────────────────────────────────

FITBYTE_CSS = """
/* ── Base ── */
.gradio-container { font-family: 'Inter', sans-serif !important; }

/* ── Header ── */
.fb-header {
    background: linear-gradient(135deg, #1A56DB 0%, #0F3D9E 100%);
    border-radius: 14px;
    padding: 28px 32px 20px;
    margin-bottom: 4px;
    color: white;
}
.fb-header h1 {
    font-size: 28px;
    font-weight: 700;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.fb-header p {
    font-size: 15px;
    opacity: 0.85;
    margin: 0;
}

/* ── Section labels ── */
.fb-section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 6px;
}

/* ── Buttons ── */
.btn-generate { background: #1A56DB !important; color: white !important; font-weight: 600 !important; border-radius: 8px !important; }
.btn-generate:hover { background: #1447C0 !important; }
.btn-refine { background: #F59E0B !important; color: white !important; font-weight: 600 !important; border-radius: 8px !important; }
.btn-refine:hover { background: #D97706 !important; }
.btn-approve { background: #10B981 !important; color: white !important; font-weight: 600 !important; border-radius: 8px !important; font-size: 15px !important; padding: 12px 28px !important; }
.btn-approve:hover { background: #059669 !important; }

/* ── Word count badge ── */
.wc-badge {
    display: inline-block;
    background: #EFF6FF;
    color: #1A56DB;
    border: 1px solid #BFDBFE;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 13px;
    font-weight: 600;
}

/* ── Output box ── */
.output-textarea textarea {
    font-size: 15px !important;
    line-height: 1.7 !important;
    border-radius: 10px !important;
    border: 1.5px solid #E5E7EB !important;
    background: white !important;
    color: #000000 !important;
}
.output-textarea textarea:focus {
    border-color: #1A56DB !important;
    background: white !important;
    color: #000000 !important;
    box-shadow: 0 0 0 3px rgba(26,86,219,0.08) !important;
}
.output-textarea textarea::placeholder {
    color: #999999 !important;
}

/* ── Refinement box ── */
.refine-box {
    background: #FFFBEB;
    border: 1.5px solid #FDE68A;
    border-radius: 10px;
    padding: 16px;
}

/* ── Approve box ── */
.approve-box {
    background: #F0FDF4;
    border: 1.5px solid #BBF7D0;
    border-radius: 10px;
    padding: 16px;
}

/* ── Tips ── */
.tip-box {
    background: #EFF6FF;
    border-left: 4px solid #1A56DB;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #1E40AF;
    line-height: 1.6;
}
"""

# ── UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="FitByte Content Creator") as demo:
    demo.queue()

    # ── Header ──────────────────────────────────────────────────────
    gr.HTML("""
    <div class="fb-header">
        <h1>FitByte AI Content Creator</h1>
        <p>Generate on-brand blog posts, captions and more — powered by your knowledge bases</p>
    </div>
    """)

    with gr.Row():

        # ── LEFT COLUMN: Inputs ──────────────────────────────────────
        with gr.Column(scale=2, min_width=320):
            gr.HTML('<div class="fb-section-label">Content Brief</div>')

            topic_input = gr.Textbox(
                label="Topic / Angle",
                placeholder='e.g. "why rest days make you fitter" or "HRV explained simply"',
                lines=2,
            )

            with gr.Row():
                audience_input = gr.Dropdown(
                    label="Target Audience",
                    choices=list(AUDIENCES.keys()),
                    value="Fitness Enthusiast",
                )
                channel_input = gr.Dropdown(
                    label="Channel",
                    choices=list(CHANNELS.keys()),
                    value="Blog Post",
                )

            custom_input = gr.Textbox(
                label="Custom Instructions (optional)",
                placeholder='e.g. "mention the FitByte Pro specifically" or "include a winter training angle"',
                lines=2,
            )

            generate_btn = gr.Button(
                "✦  Generate Content",
                variant="primary",
                elem_classes=["btn-generate"],
                size="lg",
            )

            gr.HTML("""
            <div class="tip-box" style="margin-top:12px">
                <strong>Tips for better outputs:</strong><br>
                • Be specific with your topic — "how dual-frequency GPS works" beats "GPS"<br>
                • Use Custom Instructions to pin a product model or seasonal angle<br>
                • The pipeline runs: Knowledge Base → Brief → Post (~15 seconds)
            </div>
            """)

        # ── RIGHT COLUMN: Output ─────────────────────────────────────
        with gr.Column(scale=3, min_width=400):
            gr.HTML('<div class="fb-section-label">Generated Post — edit directly in the box</div>')

            output_box = gr.Textbox(
                label="",
                lines=18,
                max_lines=30,
                placeholder="Your blog post will appear here. You can edit it directly before approving.",
                elem_classes=["output-textarea"],
                interactive=True,
            )

            word_count_display = gr.Markdown("", elem_id="wc")

            # ── Refinement row (hidden until content generated) ──
            with gr.Group(visible=False, elem_classes=["refine-box"]) as refine_row:
                gr.HTML('<div class="fb-section-label" style="color:#92400E">✎ Refine with AI</div>')
                with gr.Row():
                    refine_input = gr.Textbox(
                        label="",
                        placeholder='e.g. "make the opening punchier" · "mention the 14-day battery" · "shorten to 200 words"',
                        lines=1,
                        scale=4,
                        show_label=False,
                    )
                    refine_btn = gr.Button(
                        "Refine",
                        elem_classes=["btn-refine"],
                        scale=1,
                        min_width=90,
                    )

            # ── Approve row (hidden until content generated) ──
            with gr.Group(visible=False, elem_classes=["approve-box"]) as approve_row:
                gr.HTML('<div class="fb-section-label" style="color:#065F46">✓ Approve & Export</div>')
                with gr.Row():
                    approve_btn = gr.Button(
                        "✓  Approve & Download .txt",
                        elem_classes=["btn-approve"],
                        scale=2,
                    )
                    download_file = gr.File(
                        label="Download",
                        visible=True,
                        scale=1,
                        interactive=False,
                    )

    # ── EVENT WIRING ─────────────────────────────────────────────────

    # Generate button
    generate_btn.click(
        fn=generate_post,
        inputs=[topic_input, audience_input, channel_input, custom_input],
        outputs=[output_box, refine_row, approve_row, refine_input, word_count_display],
        queue=True,
    )

    # Refine button
    refine_btn.click(
        fn=refine_post,
        inputs=[output_box, refine_input, topic_input, audience_input, channel_input],
        outputs=[output_box],
        queue=True,
    )

    # Approve → download
    approve_btn.click(
        fn=approve_and_download,
        inputs=[output_box, topic_input, channel_input],
        outputs=[download_file],
        queue=False,
    )


if __name__ == "__main__":
    print("\n  FitByte AI Content Creator")
    print("  ──────────────────────────")
    print(f"  LLM: {llm.__class__.__name__} / {llm.model}")
    print("  Open: http://localhost:7860\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=FITBYTE_CSS,
        show_error=True,
    )
