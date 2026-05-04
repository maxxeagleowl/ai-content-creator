# FitByte AI Content Creator

An AI-powered content generation system for FitByte, a precision fitness watch brand.
Generates on-brand blog posts, Instagram captions, LinkedIn posts and email subject lines
by injecting brand guidelines, product specs and market research into advanced LLM prompts.

---

## Project Structure

```
fitbyte-ai-content-creator/
├── main.py                              # CLI entry point
├── fitbyte_ai_content_creator.ipynb     # Main project notebook
├── src/
│   ├── document_processor.py            # Markdown parsing & text extraction
│   ├── knowledge_base.py                # KB management (primary + secondary)
│   ├── prompt_templates.py              # Advanced prompt engineering templates
│   ├── content_pipeline.py              # Full pipeline orchestration
│   └── llm_integration.py              # OpenAI & Anthropic API wrapper
├── knowledge_base/
│   ├── primary/                         # Company-specific documents
│   │   ├── fitbyte_brand_guidelines.md
│   │   ├── fitbyte_product_specs.md
│   │   └── past_content/
│   │       └── fitbyte_content_examples.md
│   └── secondary/                       # Industry research
│       ├── market_trends.md
│       └── competitor_analysis.md
├── outputs/                             # Generated content (gitignored)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd fitbyte-ai-content-creator
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your API key(s)
```

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

At least one key is required. The system auto-detects which to use.

### 4. Run the notebook

```bash
jupyter notebook fitbyte_ai_content_creator.ipynb
```

Or use the CLI:

```bash
python main.py --topic "why rest days matter" --channel blog
```

---

## Content Pipeline

```
[1] DOCUMENT  — Load markdown from both knowledge bases
[2] MONITOR   — LLM analyzes topic for brand fit + market relevance
[3] BRIEF     — Chain-of-thought prompt generates a structured content brief
[4] PUBLISH   — Few-shot + contextual placement generates final content
[5] ITERATE   — Human review: approve / edit / regenerate
```

---

## CLI Usage

```bash
# Interactive blog post
python main.py --topic "why your resting heart rate matters" --channel blog

# Instagram caption (skip review)
python main.py --topic "winter training" --channel instagram --auto

# LinkedIn post
python main.py --topic "stress and recovery" --channel linkedin --audience health_professional

# Uniqueness comparison vs generic ChatGPT
python main.py --topic "sleep quality" --compare

# Full batch run
python main.py --batch

# Use Anthropic instead of OpenAI
python main.py --topic "overtraining" --provider anthropic
```

---

## Channels & Audiences

**Channels:** `blog` | `instagram` | `linkedin` | `email_subject`

**Audiences:** `fitness_enthusiast` | `performance_athlete` | `health_professional` | `upgrader` | `general`

---

## How It Avoids Generic AI Content

Every prompt is anchored to FitByte's proprietary context:

| Generic Prompt | FitByte Branded Prompt |
|---|---|
| No brand voice | FitByte's exact voice rules (active voice, no serial comma, em-dash style) |
| Vague product mentions | Specific product specs (14-day battery, dual-frequency GPS, 6-stage sleep) |
| Generic fitness language | FitByte vocabulary ("Recovery Score", "Body Battery", "fitness watch") |
| Standard blog format | Matched to FitByte's actual 150-250 word, second-person style |
| No examples | 10 real FitByte blog posts as few-shot style references |
| No market context | Market trends, consumer pain points, competitor positioning |

---

## VSCode Agent Configuration

See `config/vscode_agent.json` for the project's VSCode agent setup.

Recommended extensions:
- **Python** (ms-python.python)
- **Jupyter** (ms-toolsai.jupyter)
- **Python Docstring Generator** (njpwerner.autodocstring)

---

## Knowledge Base

### Primary (Company-Specific)
- `fitbyte_brand_guidelines.md` — Voice, writing rules, vocabulary, tone by channel
- `fitbyte_product_specs.md` — Full hardware and feature specifications
- `past_content/fitbyte_content_examples.md` — 10 real FitByte blog posts for style reference

### Secondary (Industry Research)
- `market_trends.md` — Consumer trends, pain points, platform notes (2024-2025)
- `competitor_analysis.md` — Positioning vs Garmin, Apple Watch, Fitbit, Whoop

To update knowledge bases: edit the `.md` files in `knowledge_base/`. No reindexing required.

---

## Dependencies

| Package | Purpose |
|---|---|
| `openai` | OpenAI GPT-4o-mini API |
| `anthropic` | Anthropic Claude API |
| `python-dotenv` | API key management |

---

## Team

Built for Ironhack Data Analytics Bootcamp — Module 2 Project
