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

import argparse
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
from content_pipeline import ContentPipeline
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

CLI_CHANNELS = list(CHANNEL_CONFIG.keys())
CLI_AUDIENCES = list(AUDIENCES.values())

BATCH_REQUESTS = [
    {
        "topic": "why rest days are part of training",
        "channel": "blog",
        "audience": "fitness_enthusiast",
    },
    {
        "topic": "the difference between sleep length and sleep quality",
        "channel": "blog",
        "audience": "health_professional",
    },
    {
        "topic": "how dual-frequency GPS changes route accuracy",
        "channel": "blog",
        "audience": "performance_athlete",
    },
    {
        "topic": "your body knows you're stressed before your brain does",
        "channel": "instagram",
        "audience": "fitness_enthusiast",
    },
    {
        "topic": "training load and injury prevention",
        "channel": "linkedin",
        "audience": "health_professional",
    },
    {
        "topic": "improve recovery with smarter insights",
        "channel": "email_subject",
        "audience": "general",
    },
]


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="FitByte AI Content Creator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:

  python app.py --topic "why your resting heart rate matters" --channel blog
  python app.py --topic "winter training" --channel instagram --auto
  python app.py --topic "sleep quality" --compare
  python app.py --batch
""",
    )
    parser.add_argument("--topic", type=str, help="Content topic or angle")
    parser.add_argument(
        "--channel",
        type=str,
        default="blog",
        choices=CLI_CHANNELS,
        help="Target channel (default: blog)",
    )
    parser.add_argument(
        "--audience",
        type=str,
        default="fitness_enthusiast",
        choices=CLI_AUDIENCES,
        help="Target audience (default: fitness_enthusiast)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "openai", "anthropic"],
        help="LLM provider (default: auto)",
    )
    parser.add_argument("--auto", action="store_true", help="Skip human review, auto-save")
    parser.add_argument("--compare", action="store_true", help="Run uniqueness comparison")
    parser.add_argument("--batch", action="store_true", help="Run pre-defined batch of posts")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")
    return parser.parse_args()


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


def render_star_rating_html(rating: int) -> str:
    rating_value = max(1, min(5, int(rating or 0)))
    filled = "".join('<span class="feedback-star-filled">★</span>' for _ in range(rating_value))
    empty = "".join('<span class="feedback-star-empty">☆</span>' for _ in range(5 - rating_value))
    return f'<div class="feedback-rating-visual" aria-label="Rating {rating_value} of 5">{filled}{empty}</div>'


def set_feedback_rating(rating: int) -> tuple:
    v = max(1, min(5, int(rating)))
    labels = ["★" if i <= v else "☆" for i in range(1, 6)]
    return (v, *labels)


def save_user_feedback(
    current_content: str,
    state: dict,
    rating: int,
    comment: str = "",
    user_name: str = "",
) -> tuple:
    """
    Save one feedback entry as an appended JSON record.

    Returns a status message plus cleared optional fields.
    """
    _reset = (3, "★", "★", "★", "☆", "☆")

    if not current_content or not current_content.strip():
        gr.Warning("Generate a post first before submitting feedback.")
        return ("", *_reset, comment, user_name)

    try:
        rating_value = int(rating)
    except (TypeError, ValueError):
        gr.Warning("Rating must be between 1 and 5.")
        return ("", *_reset, comment, user_name)

    if rating_value < 1 or rating_value > 5:
        gr.Warning("Rating must be between 1 and 5.")
        return ("", *_reset, comment, user_name)

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

    return "Feedback saved to `outputs/user_feedback.json`.", 3, "★", "★", "★", "☆", "☆", "", ""


# ── BUSINESS LOGIC ────────────────────────────────────────────────────

def generate_post(topic, audience_label, channel_label, custom_instructions):
    """
    Runs the LLM pipeline with comprehensive error handling.

    Returns
    -------
    content : str          ? output_box
    state   : dict         ? content_state  (JSON with full context for refine/approve)
    wc      : str          ? word_count_display
    """
    if not topic.strip():
        gr.Warning("??  Please enter a topic before generating.")
        return "", {}, "0 words"

    try:
        topic = topic.strip()
        if len(topic) < 3:
            gr.Warning("??  Topic must be at least 3 characters.")
            return "", {}, "0 words"

        if len(topic) > 200:
            gr.Warning("??  Topic must be under 200 characters.")
            return "", {}, "0 words"

        audience = AUDIENCES.get(audience_label, "fitness_enthusiast")
        channel = CHANNELS.get(channel_label, "blog")
        channel_config = get_channel_config(channel)
        gr.Info("?? Loading knowledge base...")
        context = get_cached_context(topic, audience)

        if not context or not any(context.values()):
            gr.Error("? Failed to load knowledge base. Please check the knowledge_base/ folder.")
            return "", {}, "0 words"

        gr.Info("?? Building prompt context...")
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

        gr.Info(f"? Generating {channel_config['label']}... (this may take 10-20 seconds)")
        response = llm.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=channel_config["temperature"],
            max_tokens=channel_config["max_tokens"],
            timeout=45,
        )

        if not response or not response.content:
            gr.Error("? LLM returned empty response. Please try again.")
            return "", {}, "0 words"

        content = response.content.strip()
        if not content:
            gr.Error("? Generated content is empty. Please try again.")
            return "", {}, "0 words"

        state = {
            "topic": topic,
            "channel": channel,
            "channel_label": channel_label,
            "audience": audience,
            "output_label": channel_config["label"],
            "brand_identity": context["brand_identity"][:600],
            "writing_rules": context["writing_rules"][:600],
            "audience_insights": context["audience_insights"][:600],
        }

        status_msg = f"? Content generated successfully. ({len(content.split())} words)"
        gr.Info(status_msg)
        return content, state, word_count(content)

    except ValueError as e:
        error_msg = f"? Validation error: {str(e)}"
        gr.Error(error_msg)
        print(f"ERROR in generate_post: {e}")
        return "", {}, "0 words"
    except RuntimeError as e:
        error_msg = f"? {str(e)[:100]}"
        gr.Error(error_msg)
        print(f"ERROR in generate_post: {e}")
        return "", {}, "0 words"
    except Exception as e:
        error_msg = f"? Unexpected error: {str(e)[:100]}"
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

        return new_content, updated_state, word_count(new_content)

    except ValueError as e:
        error_msg = f"❌ Validation error: {str(e)}"
        gr.Error(error_msg)
        print(f"ERROR in refine_post: {e}")
        return current_content, state, word_count(current_content)
    except RuntimeError as e:
        error_msg = f"❌ Refinement failed: {str(e)[:100]}"
        gr.Error(error_msg)
        print(f"ERROR in refine_post: {e}")
        return current_content, state, word_count(current_content)
    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)[:100]}"
        gr.Error(error_msg)
        print(f"ERROR in refine_post: {e}")
        return current_content, state, word_count(current_content)


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
            print(f"WARNING: Invalid GRADIO_SERVER_PORT: {env_port}, using default {start_port}")
    
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
.gradio-container { font-family: 'Inter', 'Segoe UI', sans-serif !important; background: #050D1A !important; }
body, .main { background: #050D1A !important; }

/* ── Header ── */
.fb-header {
    background: linear-gradient(135deg, #071D40 0%, #0F3380 55%, #1A56DB 100%);
    border-radius: 18px; padding: 36px 44px; margin-bottom: 20px; color: white;
    position: relative; overflow: hidden;
    display: flex; align-items: center; justify-content: space-between; gap: 24px;
    min-height: 190px;
}
.fb-header::before {
    content: ''; position: absolute; top: -80px; left: -40px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(96,165,250,0.14) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.fb-header::after {
    content: ''; position: absolute; bottom: -110px; right: 170px;
    width: 340px; height: 340px;
    background: radial-gradient(circle, rgba(147,197,253,0.08) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.fb-header-left { flex: 1; position: relative; z-index: 1; }
.fb-header-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.13); border: 1px solid rgba(255,255,255,0.22);
    border-radius: 20px; padding: 4px 14px; font-size: 11px; font-weight: 600;
    letter-spacing: 0.07em; text-transform: uppercase; margin-bottom: 14px;
    color: rgba(255,255,255,0.88);
}
.fb-header h1 {
    font-size: 34px; font-weight: 800; margin: 0 0 10px 0;
    letter-spacing: -1px; line-height: 1.1;
}
.fb-header p {
    font-size: 14.5px; opacity: 0.72; margin: 0; max-width: 440px; line-height: 1.55;
}
.fb-header-right { flex-shrink: 0; position: relative; z-index: 1; }

@keyframes watchFloat {
    0%, 100% { transform: translateY(0px) rotate(-4deg); }
    50%       { transform: translateY(-10px) rotate(-4deg); }
}
.fb-watch-svg {
    filter: drop-shadow(0 14px 30px rgba(0,0,0,0.5));
    animation: watchFloat 4.5s ease-in-out infinite;
    display: block;
}

/* ── Section labels ── */
.fb-section-label {
    font-size: 10.5px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #4A7CBF; margin-bottom: 8px;
}

/* ── Buttons ── */
.btn-generate { background: linear-gradient(135deg, #1A56DB 0%, #1447C0 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 10px !important; box-shadow: 0 4px 14px rgba(26,86,219,0.32) !important; border: none !important; transition: all 0.18s ease !important; }
.btn-generate:hover { background: linear-gradient(135deg, #1447C0 0%, #0F3D9E 100%) !important; box-shadow: 0 6px 18px rgba(26,86,219,0.42) !important; transform: translateY(-1px) !important; }
.btn-refine { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 10px !important; box-shadow: 0 4px 14px rgba(245,158,11,0.32) !important; border: none !important; transition: all 0.18s ease !important; }
.btn-refine:hover { background: linear-gradient(135deg, #D97706 0%, #B45309 100%) !important; transform: translateY(-1px) !important; }
.btn-approve { background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 10px !important; font-size: 15px !important; padding: 12px 28px !important; box-shadow: 0 4px 14px rgba(16,185,129,0.32) !important; border: none !important; transition: all 0.18s ease !important; }
.btn-approve:hover { background: linear-gradient(135deg, #059669 0%, #047857 100%) !important; transform: translateY(-1px) !important; }

/* ── Dark-blue input & dropdown fields ── */
.gradio-container input:not([type="range"]):not([type="checkbox"]):not([type="radio"]),
.gradio-container textarea {
    background: #071B33 !important;
    border: 1.5px solid #1A3A6E !important;
    color: #BFD4F0 !important;
    border-radius: 10px !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #2D507A !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: #3B82F6 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.18) !important;
}
/* Dropdown wrapper */
.gradio-container .wrap {
    background: #071B33 !important;
    border: 1.5px solid #1A3A6E !important;
    color: #BFD4F0 !important;
}
.gradio-container .wrap:focus-within { border-color: #3B82F6 !important; }
.gradio-container .wrap span, .gradio-container .wrap input { color: #BFD4F0 !important; }
/* Dropdown list */
.gradio-container ul.options, .gradio-container .options {
    background: #0B2040 !important;
    border: 1px solid #1A3A6E !important;
}
.gradio-container ul.options li, .gradio-container .options li { color: #BFD4F0 !important; }
.gradio-container ul.options li:hover, .gradio-container .options .active { background: #1A3A6E !important; }
/* Field labels */
.gradio-container label > span,
.gradio-container .label-wrap span { color: #5B8EC9 !important; }

/* ── Output textbox ── */
.output-textarea textarea {
    font-size: 15px !important; line-height: 1.7 !important;
    border-radius: 12px !important; border: 1.5px solid #1A3A6E !important;
    background: #071B33 !important; color: #BFD4F0 !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
}
.output-textarea textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.18) !important;
}
.output-textarea textarea::placeholder { color: #2D507A !important; }

/* ── Panels ── */
.refine-box  { background: #150E00; border: 2px solid #B45309; border-radius: 12px; padding: 16px; box-shadow: 0 2px 12px rgba(180,83,9,0.18); }
.approve-box { background: #001A0A; border: 2px solid #047857; border-radius: 12px; padding: 16px; box-shadow: 0 2px 12px rgba(4,120,87,0.18); }
.feedback-box {
    background: #150900; border: 2px solid #B45309;
    border-radius: 12px; padding: 16px; margin-top: 12px; width: 100%;
    box-shadow: 0 2px 12px rgba(180,83,9,0.18);
}
.feedback-rating-display { text-align: center; font-size: 32px; line-height: 1.1; letter-spacing: 0.1em; margin: 2px 0 10px; }
.feedback-rating-visual { display: flex; justify-content: center; align-items: center; gap: 2px; }
.feedback-star-filled { color: #B45309; }
.feedback-star-empty  { color: #374151; }
.feedback-star-row    { gap: 8px; justify-content: center; }
#feedback-panel .feedback-star-btn button,
#feedback-panel .feedback-star-btn button span,
#feedback-panel .feedback-star-btn button * {
    background: transparent !important; border: 0 !important; box-shadow: none !important;
    color: #B45309 !important; font-size: 36px !important; line-height: 1 !important;
    min-width: 0 !important; padding: 0 3px !important; transition: transform 0.12s ease, color 0.12s ease;
}
#feedback-panel .feedback-star-btn button:hover,
#feedback-panel .feedback-star-btn button:hover span { color: #D97706 !important; transform: scale(1.12); }
.status-box {
    background: #071B33; border: 1.5px solid #1A3A6E; border-radius: 10px;
    padding: 12px 14px; margin-top: 12px; font-size: 13px; color: #7EB3FF;
}
.tip-box {
    background: #071B33;
    border-left: 4px solid #1A56DB; border-radius: 0 10px 10px 0; padding: 14px 16px;
    font-size: 13px; color: #5B9BDB; line-height: 1.6;
    box-shadow: 0 2px 10px rgba(26,86,219,0.15);
}
"""

# ── UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="FitByte Content Creator", css=FITBYTE_CSS) as demo:

    # Shared state: carries generation context from generate → refine → approve
    content_state = gr.State({})

    gr.HTML("""
    <div class="fb-header">
        <div class="fb-header-left">
            <div class="fb-header-badge">&#9889; AI-Powered &nbsp;&middot;&nbsp; FitByte Pro</div>
            <h1>FitByte AI Content Creator</h1>
            <p>Generate on-brand blog posts, captions, LinkedIn posts and email subjects &mdash; powered by your knowledge base</p>
        </div>
        <div class="fb-header-right">
          <svg class="fb-watch-svg" width="130" height="185" viewBox="0 0 140 200" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="fbBandGrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#3B82F6"/><stop offset="100%" stop-color="#1D4ED8"/>
              </linearGradient>
              <linearGradient id="fbBodyGrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#3D4A5C"/><stop offset="100%" stop-color="#1E2A3A"/>
              </linearGradient>
              <linearGradient id="fbScreenGrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#0C1628"/><stop offset="100%" stop-color="#071020"/>
              </linearGradient>
              <linearGradient id="fbRingGrad" gradientUnits="userSpaceOnUse" x1="73" y1="122" x2="73" y2="158">
                <stop offset="0%" stop-color="#60A5FA"/><stop offset="100%" stop-color="#3B82F6"/>
              </linearGradient>
              <filter id="fbBodyShadow">
                <feDropShadow dx="1" dy="5" stdDeviation="6" flood-color="rgba(0,0,30,0.6)"/>
              </filter>
            </defs>
            <!-- Band top -->
            <rect x="48" y="0" width="44" height="42" rx="9" fill="url(#fbBandGrad)"/>
            <rect x="54" y="8"  width="32" height="2" rx="1" fill="rgba(255,255,255,0.18)"/>
            <rect x="54" y="15" width="32" height="2" rx="1" fill="rgba(255,255,255,0.18)"/>
            <rect x="54" y="22" width="32" height="2" rx="1" fill="rgba(255,255,255,0.18)"/>
            <rect x="54" y="29" width="32" height="2" rx="1" fill="rgba(255,255,255,0.18)"/>
            <!-- Watch case shadow -->
            <rect x="14" y="48" width="118" height="116" rx="25" fill="rgba(0,0,0,0.35)" transform="translate(2,5)"/>
            <!-- Watch case -->
            <rect x="14" y="48" width="118" height="116" rx="25" fill="url(#fbBodyGrad)"/>
            <rect x="15" y="49" width="116" height="114" rx="24" fill="none" stroke="rgba(255,255,255,0.09)" stroke-width="1.5"/>
            <!-- Screen -->
            <rect x="22" y="56" width="102" height="100" rx="19" fill="url(#fbScreenGrad)"/>
            <rect x="28" y="57" width="90" height="3" rx="1.5" fill="rgba(255,255,255,0.04)"/>
            <!-- Brand dot + name -->
            <circle cx="73" cy="70" r="4.5" fill="#3B82F6"/>
            <circle cx="73" cy="70" r="2.5" fill="#93C5FD"/>
            <text x="73" y="82" text-anchor="middle" font-family="Arial, sans-serif" font-size="7" font-weight="700" fill="rgba(255,255,255,0.35)" letter-spacing="2.5">FITBYTE</text>
            <!-- Time -->
            <text x="73" y="108" text-anchor="middle" font-family="'Courier New', monospace" font-size="27" font-weight="700" fill="white" letter-spacing="-0.5">9:41</text>
            <!-- Date -->
            <text x="73" y="121" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="rgba(255,255,255,0.38)" letter-spacing="0.5">MON 6 MAY 2026</text>
            <!-- Separator -->
            <line x1="34" y1="129" x2="112" y2="129" stroke="rgba(255,255,255,0.07)" stroke-width="1"/>
            <!-- Activity ring bg -->
            <circle cx="73" cy="143" r="17" fill="none" stroke="rgba(255,255,255,0.09)" stroke-width="4"/>
            <!-- Activity ring 75% — circ=106.8, 75%=80.1 -->
            <circle cx="73" cy="143" r="17" fill="none" stroke="url(#fbRingGrad)" stroke-width="4" stroke-dasharray="80 27" stroke-linecap="round" transform="rotate(-90 73 143)"/>
            <!-- Ring label -->
            <text x="73" y="140" text-anchor="middle" font-family="Arial, sans-serif" font-size="9.5" font-weight="700" fill="white">75%</text>
            <text x="73" y="150" text-anchor="middle" font-family="Arial, sans-serif" font-size="6.5" fill="rgba(255,255,255,0.4)">GOAL</text>
            <!-- Heart rate left -->
            <text x="34" y="137" text-anchor="middle" font-size="10" fill="#F87171">&#9829;</text>
            <text x="34" y="147" text-anchor="middle" font-family="Arial, sans-serif" font-size="9.5" font-weight="700" fill="white">72</text>
            <text x="34" y="156" text-anchor="middle" font-family="Arial, sans-serif" font-size="6.5" fill="rgba(255,255,255,0.38)">BPM</text>
            <!-- Steps right -->
            <text x="112" y="137" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#34D399">&#8593;</text>
            <text x="112" y="147" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" font-weight="700" fill="white">8.2k</text>
            <text x="112" y="156" text-anchor="middle" font-family="Arial, sans-serif" font-size="6.5" fill="rgba(255,255,255,0.38)">STEPS</text>
            <!-- Crown button -->
            <rect x="131" y="69" width="7" height="22" rx="3.5" fill="#2D3748"/>
            <rect x="132" y="70" width="5" height="20" rx="2.5" fill="#4A5568"/>
            <rect x="133" y="74" width="3" height="12" rx="1.5" fill="#606C7A"/>
            <!-- Band bottom -->
            <rect x="48" y="162" width="44" height="38" rx="9" fill="url(#fbBandGrad)"/>
            <rect x="54" y="166" width="32" height="2" rx="1" fill="rgba(255,255,255,0.18)"/>
            <rect x="54" y="173" width="32" height="2" rx="1" fill="rgba(255,255,255,0.18)"/>
            <rect x="54" y="180" width="32" height="2" rx="1" fill="rgba(255,255,255,0.18)"/>
            <rect x="54" y="187" width="32" height="2" rx="1" fill="rgba(255,255,255,0.18)"/>
          </svg>
        </div>
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

            gr.HTML("""
            <div class="tip-box" style="margin-top:12px">
                <strong>Tips for better outputs:</strong><br>
                • Be specific — "dual-frequency GPS" beats "GPS"<br>
                • Pin a product model or seasonal angle via Custom Instructions<br>
                • Generation takes ~10–15 s
            </div>
            """)

            generate_btn = gr.Button(
                "✦  Generate Content",
                variant="primary",
                elem_classes=["btn-generate"],
                size="lg",
            )

            # ── Feedback ────────────────────────────────────────────
            with gr.Group(elem_id="feedback-panel", elem_classes=["feedback-box"]):
                gr.HTML('<div class="fb-section-label" style="color:#9A3412">User Feedback</div>')
                feedback_status = gr.Markdown("")
                feedback_rating_state = gr.State(3)
                with gr.Row(elem_classes=["feedback-star-row"]):
                    star_1 = gr.Button("★", elem_classes=["feedback-star-btn"], variant="secondary", min_width=0)
                    star_2 = gr.Button("★", elem_classes=["feedback-star-btn"], variant="secondary", min_width=0)
                    star_3 = gr.Button("★", elem_classes=["feedback-star-btn"], variant="secondary", min_width=0)
                    star_4 = gr.Button("☆", elem_classes=["feedback-star-btn"], variant="secondary", min_width=0)
                    star_5 = gr.Button("☆", elem_classes=["feedback-star-btn"], variant="secondary", min_width=0)
                gr.HTML("""<script>
(function() {
  function paintStars() {
    document.querySelectorAll('.feedback-star-btn button, .feedback-star-btn button span').forEach(function(el) {
      el.style.setProperty('color', '#B45309', 'important');
      el.style.setProperty('background', 'transparent', 'important');
      el.style.setProperty('border', 'none', 'important');
      el.style.setProperty('box-shadow', 'none', 'important');
      el.style.setProperty('font-size', '32px', 'important');
    });
  }
  setTimeout(paintStars, 300);
  var obs = new MutationObserver(paintStars);
  obs.observe(document.body, { childList: true, subtree: true });
})();
</script>""")
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
                gr.HTML('<div class="fb-section-label" style="color:#ffffff">✓ Approve & Export</div>')
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

    _star_outputs = [feedback_rating_state, star_1, star_2, star_3, star_4, star_5]
    star_1.click(fn=lambda: set_feedback_rating(1), inputs=[], outputs=_star_outputs, queue=False)
    star_2.click(fn=lambda: set_feedback_rating(2), inputs=[], outputs=_star_outputs, queue=False)
    star_3.click(fn=lambda: set_feedback_rating(3), inputs=[], outputs=_star_outputs, queue=False)
    star_4.click(fn=lambda: set_feedback_rating(4), inputs=[], outputs=_star_outputs, queue=False)
    star_5.click(fn=lambda: set_feedback_rating(5), inputs=[], outputs=_star_outputs, queue=False)

    generate_btn.click(
        fn=generate_post,
        inputs=[topic_input, audience_input, channel_input, custom_input],
        outputs=[output_box, content_state, word_count_display],
        queue=True,
    )

    refine_btn.click(
        fn=refine_post,
        inputs=[output_box, refine_input, content_state],
        outputs=[output_box, content_state, word_count_display],
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
        inputs=[output_box, content_state, feedback_rating_state, feedback_comment, feedback_name],
        outputs=[feedback_status, feedback_rating_state, star_1, star_2, star_3, star_4, star_5, feedback_comment, feedback_name],
        queue=False,
    )


demo.queue()


def run_cli(args):
    try:
        pipeline = ContentPipeline(provider=args.provider, verbose=not args.quiet)
    except ValueError as e:
        print(f"ERROR: Configuration error: {e}")
        return
    except Exception as e:
        print(f"ERROR: Failed to initialize pipeline: {str(e)[:150]}")
        return

    try:
        if args.batch:
            if not BATCH_REQUESTS:
                print("ERROR: No batch requests defined.")
                return
            print(f"Running batch: {len(BATCH_REQUESTS)} content pieces\n")
            results = pipeline.run_batch(BATCH_REQUESTS)
            completed = len([r for r in results if not r.get("error")])
            failed = len(results) - completed
            print(f"\nDONE: Batch complete: {completed} succeeded", end="")
            if failed > 0:
                print(f", {failed} failed")
            else:
                print()
            return

        if not args.topic:
            print("Enter a content topic (or press Ctrl+C to quit):")
            try:
                args.topic = input("> ").strip()
            except KeyboardInterrupt:
                print("\nExiting.")
                return
            except EOFError:
                print("\nNo input received. Exiting.")
                return

            if not args.topic:
                print("ERROR: No topic provided. Exiting.")
                return

        if args.compare:
            try:
                pipeline.compare_uniqueness(topic=args.topic, channel=args.channel)
            except Exception as e:
                print(f"ERROR: Comparison failed: {str(e)[:150]}")
            return

        result = pipeline.run(
            topic=args.topic,
            channel=args.channel,
            audience=args.audience,
            auto_approve=args.auto,
        )

        if result.get("error"):
            print(f"\nERROR: Generation failed: {result['error']}")
            return

        if result.get("content") and result["content"] not in ("", "__REGENERATE__"):
            print("\nDONE: Content saved to outputs/")
        else:
            print("\nWARNING: No content generated.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nERROR: Unexpected error: {str(e)[:150]}")


def run_ui():
    try:
        server_port = find_free_port()
        print("\n=== FitByte AI Content Creator ===")
        print("==================================")
        print(f"LLM: {llm.__class__.__name__} / {llm.model}")
        print(f"Open: http://localhost:{server_port}\n")
        demo.launch(
            server_name="0.0.0.0",
            server_port=server_port,
            share=False,
            inbrowser=True,
            show_error=True,
        )
    except OSError as e:
        print(f"\nERROR: Failed to start server: {e}")
        print("Try: export GRADIO_SERVER_PORT=7861 (or another free port)")
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli(parse_cli_args())
    else:
        run_ui()

