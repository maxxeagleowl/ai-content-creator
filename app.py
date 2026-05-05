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
import json
import tempfile
import socket
from threading import Lock
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
    build_writer_system_prompt,
    get_prompt_for_channel,
)

# ── INIT ──────────────────────────────────────────────────────────────
kb  = KnowledgeBase()
llm = get_llm_client(provider="auto")

_kb_cache: dict = {}
_feedback_lock = Lock()
FEEDBACK_FILE = Path(__file__).parent / "outputs" / "user_feedback.json"

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

CHANNEL_CONFIG = {
    "blog": {
        "label": "blog post",
        "max_tokens": 700,
        "temperature": 0.75,
    },
    "instagram": {
        "label": "Instagram caption",
        "max_tokens": 180,
        "temperature": 0.8,
    },
    "linkedin": {
        "label": "LinkedIn post",
        "max_tokens": 260,
        "temperature": 0.7,
    },
    "email_subject": {
        "label": "email subject line set",
        "max_tokens": 140,
        "temperature": 0.65,
    },
}


# ── HELPERS ───────────────────────────────────────────────────────────

def word_count(text: str) -> str:
    if not text:
        return "0 words"
    return f"{len(text.split())} words"


def get_channel_config(channel: str) -> dict:
    return CHANNEL_CONFIG.get(channel, CHANNEL_CONFIG["blog"])


def build_refine_prompt(current_content: str, refinement_instruction: str, state: dict) -> str:
    channel = state.get("channel", "blog") if state else "blog"
    channel_config = get_channel_config(channel)
    brand_rules = state.get("writing_rules", "")[:400] if state else ""
    brand_identity = state.get("brand_identity", "")[:400] if state else ""
    audience_insights = state.get("audience_insights", "")[:400] if state else ""

    return f"""You are editing an existing FitByte {channel_config['label']}. Apply ONLY the instruction below — keep everything else the same.

CURRENT CONTENT:
{current_content}

INSTRUCTION TO APPLY:
{refinement_instruction.strip()}

BRAND RULES (do not break these):
{brand_rules}

BRAND IDENTITY:
{brand_identity}

AUDIENCE INSIGHTS:
{audience_insights}

Return ONLY the revised content. Do not add a preamble, notes, or explanation."""


def save_user_feedback(
    current_content: str,
    state: dict,
    rating: int,
    comment: str = "",
    user_name: str = "",
) -> tuple[str, str, str]:
    """
    Save one feedback entry as an appended JSON record.

    Returns a status message plus cleared optional fields.
    """
    if not current_content or not current_content.strip():
        gr.Warning("Generate a post first before submitting feedback.")
        return "", comment, user_name

    try:
        rating_value = int(rating)
    except (TypeError, ValueError):
        gr.Warning("Rating must be between 1 and 5.")
        return "", comment, user_name

    if rating_value < 1 or rating_value > 5:
        gr.Warning("Rating must be between 1 and 5.")
        return "", comment, user_name

    entry = {
        "feedback_id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "rating": rating_value,
        "comment": comment.strip() or None,
        "user_name": user_name.strip() or None,
        "output_text": current_content.strip(),
        "output_excerpt": current_content.strip()[:500],
        "output_word_count": len(current_content.split()),
        "topic": (state or {}).get("topic"),
        "channel": (state or {}).get("channel"),
        "channel_label": (state or {}).get("channel_label"),
        "audience": (state or {}).get("audience"),
        "output_label": (state or {}).get("output_label"),
    }

    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

    with _feedback_lock:
        existing = []
        if FEEDBACK_FILE.exists():
            try:
                with FEEDBACK_FILE.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, list):
                    existing = loaded
                elif isinstance(loaded, dict):
                    existing = [loaded]
            except json.JSONDecodeError:
                existing = []

        existing.append(entry)
        with FEEDBACK_FILE.open("w", encoding="utf-8") as handle:
            json.dump(existing, handle, ensure_ascii=False, indent=2)

    return "Feedback saved to `outputs/user_feedback.json`.", "", ""


# ── BUSINESS LOGIC ────────────────────────────────────────────────────

def generate_post(topic, audience_label, channel_label, custom_instructions):
    """
    Runs the LLM pipeline with comprehensive error handling.

    Returns
    -------
    content : str          → output_box
    state   : dict         → content_state  (JSON with full context for refine/approve)
    wc      : str          → word_count_display
    """
    if not topic.strip():
        gr.Warning("⚠️  Please enter a topic before generating.")
        return "", {}, "0 words"

    try:
        # Validate inputs
        topic = topic.strip()
        if len(topic) < 3:
            gr.Warning("⚠️  Topic must be at least 3 characters.")
            return "", {}, "0 words"
        
        if len(topic) > 200:
            gr.Warning("⚠️  Topic must be under 200 characters.")
            return "", {}, "0 words"

        audience = AUDIENCES.get(audience_label, "fitness_enthusiast")
        channel = CHANNELS.get(channel_label, "blog")
        channel_config = get_channel_config(channel)
        
        gr.Info("🔄 Loading knowledge base...")
        context  = get_cached_context(topic, audience)
        
        # Validate context loaded properly
        if not context or not any(context.values()):
            gr.Error("❌ Failed to load knowledge base. Please check the knowledge_base/ folder.")
            return "", {}, "0 words"

        gr.Info("💭 Building prompt context...")
        ctx = PromptContext(
            topic=topic,
            channel=channel,
            audience=audience,
            brand_identity=context["brand_identity"],
            brand_voice=context["brand_voice"],
            writing_rules=context["writing_rules"],
            content_examples=context["content_examples"],
            product_specs=context["product_specs"],
            market_context=context["market_context"],
            differentiators=context["differentiators"],
            audience_insights=context["audience_insights"],
            extra_instructions=custom_instructions or "",
        )

        system_prompt, user_prompt = get_prompt_for_channel(ctx, brief="")

        if custom_instructions.strip():
            user_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{custom_instructions.strip()}"

        gr.Info(f"⏳ Generating {channel_config['label']}... (this may take 10-20 seconds)")
        response = llm.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=channel_config["temperature"],
            max_tokens=channel_config["max_tokens"],
            timeout=45,
        )

        if not response or not response.content:
            gr.Error("❌ LLM returned empty response. Please try again.")
            return "", {}, "0 words"

        content = response.content.strip()
        if not content:
            gr.Error("❌ Generated content is empty. Please try again.")
            return "", {}, "0 words"

        state = {
            "topic":         topic,
            "channel":       channel,
            "channel_label": channel_label,
            "audience":      audience,
            "output_label":  channel_config["label"],
            "brand_identity": context["brand_identity"][:600],
            "writing_rules": context["writing_rules"][:600],
            "audience_insights": context["audience_insights"][:600],
        }

        gr.Info(f"✅ Content generated! ({len(content.split())} words)")
        return content, state, word_count(content)

    except ValueError as e:
        error_msg = f"❌ Validation error: {str(e)}"
        gr.Error(error_msg)
        print(f"ERROR in generate_post: {e}")
        return "", {}, "0 words"
    except RuntimeError as e:
        error_msg = f"❌ {str(e)[:100]}"
        gr.Error(error_msg)
        print(f"ERROR in generate_post: {e}")
        return "", {}, "0 words"
    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)[:100]}"
        gr.Error(error_msg)
        print(f"ERROR in generate_post: {e}")
        return "", {}, "0 words"


def refine_post(current_content, refinement_instruction, state):
    """
    Refines the current post with error handling and user feedback.

    Inputs  : current_content (output_box), refinement_instruction, state (content_state)
    Returns : new_content → output_box  |  updated_state → content_state
    """
    if not current_content.strip():
        gr.Warning("⚠️  Generate a post first before refining.")
        return current_content, state

    if not refinement_instruction.strip():
        gr.Warning("⚠️  Enter a refinement instruction (e.g., 'make it shorter' or 'add more emojis').")
        return current_content, state

    if not state or not isinstance(state, dict):
        gr.Warning("⚠️  No generation context found — please generate a post first.")
        return current_content, state

    try:
        # Validate instruction length
        if len(refinement_instruction) > 300:
            gr.Warning("⚠️  Refinement instruction is too long (max 300 characters).")
            return current_content, state

        refine_prompt = build_refine_prompt(current_content, refinement_instruction, state)
        
        gr.Info("🔄 Applying refinement...")
        response = llm.generate(
            user_prompt=refine_prompt,
            system_prompt=build_writer_system_prompt(
                PromptContext(
                    topic=state.get("topic", ""),
                    channel=state.get("channel", "blog"),
                    audience=state.get("audience", "fitness_enthusiast"),
                    brand_identity=state.get("brand_identity", ""),
                    audience_insights=state.get("audience_insights", ""),
                )
            ),
            temperature=0.65,
            max_tokens=600,
            timeout=45,
        )

        if not response or not response.content.strip():
            gr.Error("❌ Failed to refine post. Please try again or regenerate from scratch.")
            return current_content, state

        new_content   = response.content.strip()
        updated_state = {**state, "content": new_content}
        
        old_wc = len(current_content.split())
        new_wc = len(new_content.split())
        gr.Info(f"✅ Refinement applied ({old_wc} → {new_wc} words)")
        
        return new_content, updated_state

    except ValueError as e:
        error_msg = f"❌ Validation error: {str(e)}"
        gr.Error(error_msg)
        print(f"ERROR in refine_post: {e}")
        return current_content, state
    except RuntimeError as e:
        error_msg = f"❌ Refinement failed: {str(e)[:100]}"
        gr.Error(error_msg)
        print(f"ERROR in refine_post: {e}")
        return current_content, state
    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)[:100]}"
        gr.Error(error_msg)
        print(f"ERROR in refine_post: {e}")
        return current_content, state


def approve_and_download(current_content, state):
    """
    Saves the approved post as a .txt file with error handling and user feedback.

    Inputs  : current_content (output_box), state (content_state)
    Returns : file path → download_file
    """
    content = current_content.strip()
    if not content:
        gr.Warning("⚠️  Nothing to download — generate a post first.")
        return None

    try:
        # Validate content
        if len(content) < 5:
            gr.Warning("⚠️  Content is too short to save.")
            return None

        topic         = state.get("topic", "post") if state else "post"
        channel_label = state.get("channel_label", "Blog Post") if state else "Blog Post"
        channel       = state.get("channel", "blog") if state else "blog"

        # Validate state
        if not state or not isinstance(state, dict):
            gr.Warning("⚠️  No state information available.")
            return None

        try:
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

            gr.Info(f"✅ File saved successfully! ({len(content.split())} words)")
            print(f"  Saved to: {tmp.name}")
            return tmp.name

        except IOError as e:
            raise RuntimeError(f"Failed to write file: {e}")
            
    except IOError as e:
        error_msg = f"❌ File I/O error: {str(e)[:100]}"
        gr.Error(error_msg)
        print(f"ERROR in approve_and_download: {e}")
        return None
    except ValueError as e:
        error_msg = f"❌ Invalid data: {str(e)[:100]}"
        gr.Error(error_msg)
        print(f"ERROR in approve_and_download: {e}")
        return None
    except Exception as e:
        error_msg = f"❌ Failed to save file: {str(e)[:100]}"
        gr.Error(error_msg)
        print(f"ERROR in approve_and_download: {e}")
        return None


# ── PORT HELPER ───────────────────────────────────────────────────────

def find_free_port(start_port=7860, max_attempts=25):
    """Find a free port with error handling."""
    env_port = os.getenv("GRADIO_SERVER_PORT")
    if env_port:
        try:
            start_port = int(env_port)
        except ValueError:
            print(f"  ⚠ Invalid GRADIO_SERVER_PORT: {env_port}, using default {start_port}")
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    
    raise OSError(f"✗ No open port found in {start_port}–{start_port + max_attempts - 1}.")


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
.feedback-box {
    background: #FFF7ED;
    border: 1.5px solid #FDBA74;
    border-radius: 10px;
    padding: 16px;
    margin-top: 12px;
    max-width: 420px;
    margin-left: auto;
}
#feedback-panel {
    position: fixed;
    right: 24px;
    bottom: 24px;
    width: 320px;
    z-index: 50;
}
@media (max-width: 900px) {
    #feedback-panel {
        position: static;
        width: 100%;
        right: auto;
        bottom: auto;
        z-index: auto;
        margin-top: 16px;
    }
}
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
        <p>Generate on-brand blog posts, captions, LinkedIn posts and email subjects — powered by your knowledge bases</p>
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
            gr.HTML('<div class="fb-section-label">Generated Content — edit directly in the box</div>')

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

    with gr.Group(elem_id="feedback-panel", elem_classes=["feedback-box"]):
        gr.HTML('<div class="fb-section-label" style="color:#9A3412">User Feedback</div>')
        feedback_status = gr.Markdown("")
        feedback_rating = gr.Slider(
            minimum=1,
            maximum=5,
            value=3,
            step=1,
            label="Rating",
        )
        feedback_comment = gr.Textbox(
            label="Comment (optional)",
            placeholder="What should be improved?",
            lines=2,
        )
        feedback_name = gr.Textbox(
            label="User Name (optional)",
            placeholder="Your name",
            lines=1,
        )
        feedback_submit_btn = gr.Button(
            "Submit Feedback",
            elem_classes=["btn-generate"],
            size="sm",
        )

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

    feedback_submit_btn.click(
        fn=save_user_feedback,
        inputs=[output_box, content_state, feedback_rating, feedback_comment, feedback_name],
        outputs=[feedback_status, feedback_comment, feedback_name],
        queue=False,
    )


demo.queue()

if __name__ == "__main__":
    try:
        server_port = find_free_port()
        print("\n  ✅ FitByte AI Content Creator")
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
    except OSError as e:
        print(f"\n✗ Failed to start server: {e}")
        print("  Try: export GRADIO_SERVER_PORT=7861 (or another free port)")
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")





