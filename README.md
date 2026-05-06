# FitByte AI Content Creator

An AI-powered content generation system for FitByte, a precision fitness watch brand.
Generates on-brand blog posts, Instagram captions, LinkedIn posts and email subject lines
by injecting brand guidelines, product specs and market research into advanced LLM prompts.

---

## Project Structure

```
fitbyte-ai-content-creator/
â”œâ”€â”€ app.py                              # CLI entry point
â”œâ”€â”€ fitbyte_ai_content_creator.ipynb     # Main project notebook
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ document_processor.py            # Markdown parsing & text extraction
â”‚   â”œâ”€â”€ knowledge_base.py                # KB management (primary + secondary)
â”‚   â”œâ”€â”€ prompt_templates.py              # Advanced prompt engineering templates
â”‚   â”œâ”€â”€ content_pipeline.py              # Full pipeline orchestration
â”‚   â””â”€â”€ llm_integration.py              # OpenAI & Anthropic API wrapper
â”œâ”€â”€ knowledge_base/
â”‚   â”œâ”€â”€ primary/                         # Company-specific documents
â”‚   â”‚   â”œâ”€â”€ fitbyte_brand_guidelines.md
â”‚   â”‚   â”œâ”€â”€ fitbyte_product_specs.md
â”‚   â”‚   â””â”€â”€ past_content/
â”‚   â”‚       â””â”€â”€ fitbyte_content_examples.md
â”‚   â””â”€â”€ secondary/                       # Industry research
â”‚       â”œâ”€â”€ market_trends.md
â”‚       â””â”€â”€ competitor_analysis.md
â”œâ”€â”€ outputs/                             # Generated content (gitignored)
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ .env.example
â””â”€â”€ .gitignore
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
python app.py --topic "why rest days matter" --channel blog
```

---

## Content Pipeline

```
[1] DOCUMENT  â€” Load markdown from both knowledge bases
[2] MONITOR   â€” LLM analyzes topic for brand fit + market relevance
[3] BRIEF     â€” Chain-of-thought prompt generates a structured content brief
[4] PUBLISH   â€” Few-shot + contextual placement generates final content
[5] ITERATE   â€” Human review: approve / edit / regenerate
```

---

## CLI Usage

```bash
# Interactive blog post
python app.py --topic "why your resting heart rate matters" --channel blog

# Instagram caption (skip review)
python app.py --topic "winter training" --channel instagram --auto

# LinkedIn post
python app.py --topic "stress and recovery" --channel linkedin --audience health_professional

# Uniqueness comparison vs generic ChatGPT
python app.py --topic "sleep quality" --compare

# Full batch run
python app.py --batch

# Use Anthropic instead of OpenAI
python app.py --topic "overtraining" --provider anthropic
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
- `fitbyte_brand_guidelines.md` â€” Voice, writing rules, vocabulary, tone by channel
- `fitbyte_product_specs.md` â€” Full hardware and feature specifications
- `past_content/fitbyte_content_examples.md` â€” 10 real FitByte blog posts for style reference

### Secondary (Industry Research)
- `market_trends.md` â€” Consumer trends, pain points, platform notes (2024-2025)
- `competitor_analysis.md` â€” Positioning vs Garmin, Apple Watch, Fitbit, Whoop

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

Built for Ironhack Data Analytics Bootcamp â€” Module 2 Project

