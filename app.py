"""
app.py
------
FitByte AI Content Creator — Gradio Interface

Workflow:
  1. User fills in topic, audience, channel, and optional custom instructions
  2. Click "Generate" → LLM runs, result appears in the output box
  3. A gr.State JSON holds all context (topic, channel, audience, writing_rules)
  4. "Refine with AI" reads from the State — no need to pass inputs again
  5. "Approve & Download" saves the final post as a .txt file

Run with:
    python app.py
"""

import sys
import os
import tempfile
import socket
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

import gradio as gr
from knowledge_base import KnowledgeBase
from llm_integration import get_llm_client
from prompt_templates import (
    PromptContext,
    build_blog_post_prompt,
    build_writer_system_prompt,
)

# ── INIT ──────────────────────────────────────────────────────────────
kb  = KnowledgeBase()
llm = get_llm_client(provider="auto")

_kb_cache: dict = {}

def get_cached_context(topic: str, audience: str) -> dict:
    key = f"{topic}:{audience}"
    if key not in _kb_cache:
        _kb_cache[key] = kb.full_context_for_generation(topic=topic, audience=audience)
    return _kb_cache[key]

AUDIENCES = {
    "Fitness Enthusiast":                   "fitness_enthusiast",
    "Performance Athlete":                  "performance_athlete",
    "Health-Conscious Professional":        "health_professional",
    "Upgrader (switching from another watch)": "upgrader",
    "General":                              "general",
}

CHANNELS = {
    "Blog Post":            "blog",
    "Instagram Caption":    "instagram",
    "LinkedIn Post":        "linkedin",
    "Email Subject Lines":  "email_subject",
}


# ── HELPERS ───────────────────────────────────────────────────────────

def word_count(text: str) -> str:
    if not text:
        return "0 words"
    return f"{len(text.split())} words"


# ── BUSINESS LOGIC ────────────────────────────────────────────────────

def generate_post(topic, audience_label, channel_label, custom_instructions):
    """
    Runs the LLM pipeline.

    Returns
    -------
    content : str          → output_box
    state   : dict         → content_state  (JSON with full context for refine/approve)
    wc      : str          → word_count_display
    """
    if not topic.strip():
        gr.Warning("Please enter a topic before generating.")
        return "", {}, "0 words"

    try:
        audience = AUDIENCES.get(audience_label, "fitness_enthusiast")
        channel  = CHANNELS.get(channel_label, "blog")
        context  = get_cached_context(topic, audience)

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

        blog_prompt = build_blog_post_prompt(ctx, brief="")
        if custom_instructions.strip():
            blog_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{custom_instructions.strip()}"

        response = llm.generate(
            user_prompt=blog_prompt,
            system_prompt=build_writer_system_prompt(ctx),
            temperature=0.75,
            max_tokens=500,
            timeout=45,
        )

        content = response.content.strip()
        if not content:
            gr.Error("Failed to generate content. Please try again.")
            return "", {}, "0 words"

        state = {
            "topic":         topic,
            "channel":       channel,
            "channel_label": channel_label,
            "audience":      audience,
            "writing_rules": context["writing_rules"][:600],
        }

        return content, state, word_count(content)

    except Exception as e:
        print(f"ERROR in generate_post: {e}")
        gr.Error(f"Error: {str(e)[:120]}")
        return "", {}, "0 words"


def refine_post(current_content, refinement_instruction, state):
    """
    Refines the current post.

    Inputs  : current_content (output_box), refinement_instruction, state (content_state)
    Returns : new_content → output_box  |  updated_state → content_state
    """
    if not current_content.strip():
        gr.Warning("Generate a post first before refining.")
        return current_content, state

    if not refinement_instruction.strip():
        gr.Warning("Enter a refinement instruction.")
        return current_content, state

    if not state:
        gr.Warning("No generation context found — please generate a post first.")
        return current_content, state

    try:
        refine_prompt = f"""You are editing an existing FitByte blog post. Apply ONLY the instruction below — keep everything else the same.

CURRENT POST:
{current_content}

INSTRUCTION TO APPLY:
{refinement_instruction.strip()}

BRAND RULES (do not break these):
{state.get('writing_rules', '')[:400]}

Return ONLY the revised post — no preamble, no notes, no explanation."""

        response = llm.generate(
            user_prompt=refine_prompt,
            system_prompt=build_writer_system_prompt(
                PromptContext(
                    topic=state.get("topic", ""),
                    channel=state.get("channel", "blog"),
                    audience=state.get("audience", "fitness_enthusiast"),
                )
            ),
            temperature=0.65,
            max_tokens=600,
            timeout=45,
        )

        if not response or not response.content.strip():
            gr.Error("Failed to refine post. Please try again.")
            return current_content, state

        new_content   = response.content.strip()
        updated_state = {**state, "content": new_content}
        return new_content, updated_state

    except Exception as e:
        print(f"ERROR in refine_post: {e}")
        gr.Error(f"Error refining post: {str(e)[:120]}")
        return current_content, state


def approve_and_download(current_content, state):
    """
    Saves the approved post as a .txt file.

    Inputs  : current_content (output_box), state (content_state)
    Returns : file path → download_file
    """
    content = current_content.strip()
    if not content:
        gr.Warning("Nothing to download — generate a post first.")
        return None

    try:
        topic         = state.get("topic", "post") if state else "post"
        channel_label = state.get("channel_label", "Blog Post") if state else "Blog Post"
        channel       = state.get("channel", "blog") if state else "blog"

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False,
            prefix=f"fitbyte_{channel}_"
        )
        tmp.write("FitByte Content — Approved\n")
        tmp.write(f"Topic:   {topic}\n")
        tmp.write(f"Channel: {channel_label}\n")
        tmp.write(f"Date:    {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        tmp.write("\n" + "=" * 60 + "\n\n")
        tmp.write(content)
        tmp.close()

        gr.Info("File saved successfully!")
        return tmp.name

    except Exception as e:
        print(f"ERROR in approve_and_download: {e}")
        gr.Error(f"Error saving file: {str(e)[:120]}")
        return None


# ── PORT HELPER ───────────────────────────────────────────────────────

def find_free_port(start_port=7860, max_attempts=25):
    env_port = os.getenv("GRADIO_SERVER_PORT")
    if env_port:
        try:
            start_port = int(env_port)
        except ValueError:
            pass
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise OSError(f"No open port found in {start_port}–{start_port + max_attempts - 1}.")


# ── CSS ───────────────────────────────────────────────────────────────

FITBYTE_CSS = """
.gradio-container { font-family: 'Inter', sans-serif !important; }

.fb-header {
    background: linear-gradient(135deg, #1A56DB 0%, #0F3D9E 100%);
    border-radius: 14px; padding: 28px 32px 20px; margin-bottom: 4px; color: white;
}
.fb-header h1 { font-size: 28px; font-weight: 700; margin: 0 0 4px 0; letter-spacing: -0.5px; }
.fb-header p  { font-size: 15px; opacity: 0.85; margin: 0; }

.fb-section-label {
    font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #6B7280; margin-bottom: 6px;
}

.btn-generate { background: #1A56DB !important; color: white !important; font-weight: 600 !important; border-radius: 8px !important; }
.btn-generate:hover { background: #1447C0 !important; }
.btn-refine   { background: #F59E0B !important; color: white !important; font-weight: 600 !important; border-radius: 8px !important; }
.btn-refine:hover { background: #D97706 !important; }
.btn-approve  { background: #10B981 !important; color: white !important; font-weight: 600 !important; border-radius: 8px !important; font-size: 15px !important; padding: 12px 28px !important; }
.btn-approve:hover { background: #059669 !important; }

.output-textarea textarea {
    font-size: 15px !important; line-height: 1.7 !important;
    border-radius: 10px !important; border: 1.5px solid #E5E7EB !important;
    background: white !important; color: #000000 !important;
}
.output-textarea textarea:focus {
    border-color: #1A56DB !important; background: white !important; color: #000000 !important;
    box-shadow: 0 0 0 3px rgba(26,86,219,0.08) !important;
}
.output-textarea textarea::placeholder { color: #999999 !important; }

.refine-box  { background: #FFFBEB; border: 1.5px solid #FDE68A; border-radius: 10px; padding: 16px; }
.approve-box { background: #F0FDF4; border: 1.5px solid #BBF7D0; border-radius: 10px; padding: 16px; }
.tip-box {
    background: #EFF6FF; border-left: 4px solid #1A56DB;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    font-size: 13px; color: #1E40AF; line-height: 1.6;
}
"""

# ── UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="FitByte Content Creator", css=FITBYTE_CSS) as demo:

    # Shared state: carries generation context from generate → refine → approve
    content_state = gr.State({})

    gr.HTML("""
    <div class="fb-header">
        <h1>FitByte AI Content Creator</h1>
        <p>Generate on-brand blog posts, captions and more — powered by your knowledge bases</p>
    </div>
    """)

    with gr.Row():

        # ── LEFT: Inputs ─────────────────────────────────────────────
        with gr.Column(scale=2, min_width=320):
            gr.HTML('<div class="fb-section-label">Content Brief</div>')

            topic_input = gr.Textbox(
                label="Topic / Angle",
                placeholder='"why rest days make you fitter" or "HRV explained simply"',
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
                placeholder='"mention the FitByte Pro specifically" or "include a winter training angle"',
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
                • Be specific — "dual-frequency GPS" beats "GPS"<br>
                • Pin a product model or seasonal angle via Custom Instructions<br>
                • Generation takes ~10–15 s
            </div>
            """)

        # ── RIGHT: Output + Refine + Approve ─────────────────────────
        with gr.Column(scale=3, min_width=400):
            gr.HTML('<div class="fb-section-label">Generated Post — edit directly in the box</div>')

            output_box = gr.Textbox(
                label="",
                lines=18,
                max_lines=30,
                placeholder="Your post will appear here. You can edit it directly before approving.",
                elem_classes=["output-textarea"],
                interactive=True,
            )

            word_count_display = gr.Markdown("", elem_id="wc")

            # ── Refine ──────────────────────────────────────────────
            with gr.Group(elem_classes=["refine-box"]):
                gr.HTML('<div class="fb-section-label" style="color:#92400E">✎ Refine with AI</div>')
                with gr.Row():
                    refine_input = gr.Textbox(
                        label="",
                        placeholder='"make the opening punchier" · "mention 14-day battery" · "shorten to 200 words"',
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

            # ── Approve ─────────────────────────────────────────────
            with gr.Group(elem_classes=["approve-box"]):
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

    generate_btn.click(
        fn=generate_post,
        inputs=[topic_input, audience_input, channel_input, custom_input],
        outputs=[output_box, content_state, word_count_display],
        queue=True,
    )

    refine_btn.click(
        fn=refine_post,
        inputs=[output_box, refine_input, content_state],
        outputs=[output_box, content_state],
        queue=True,
    )

    approve_btn.click(
        fn=approve_and_download,
        inputs=[output_box, content_state],
        outputs=[download_file],
        queue=False,
    )


demo.queue()

if __name__ == "__main__":
    server_port = find_free_port()
    print("\n  FitByte AI Content Creator")
    print("  ──────────────────────────")
    print(f"  LLM: {llm.__class__.__name__} / {llm.model}")
    print(f"  Open: http://localhost:{server_port}\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=False,
        inbrowser=True,
        show_error=True,
    )
