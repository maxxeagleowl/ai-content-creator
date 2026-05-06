# FitByte AI Content Creator

Generate on-brand blog posts, Instagram captions, LinkedIn posts and email subject lines for FitByte — a precision fitness watch brand — in seconds.

Every output is grounded in FitByte's actual brand voice, real product specs, and past content. Not generic AI text.

---

## What it does

You pick a topic, a target audience and a channel. The system pulls from two knowledge bases — brand guidelines, product specs, market research — and generates content that sounds like FitByte wrote it.

You can then refine it with a plain-language instruction ("make it shorter", "add the HRV angle", "tie it to Ironman season"), approve it, and download it as a `.txt` file.

---

## Quick start

### 1. Clone the repo and install dependencies

```bash
git clone <your-repo-url>
cd ai-content-creator
pip install -r requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env
```

Open `.env` and add at least one key:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

The system auto-detects which key is available. Either works.

### 3. Launch the web app

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## Using the web app

The Gradio interface has three main areas:

**Left column — Content Brief**
- **Topic / Angle** — what the post is about. The more specific, the better ("dual-frequency GPS for trail runners" beats "GPS").
- **Target Audience** — who you're writing for. Options: Fitness Enthusiast, Performance Athlete, Health-Conscious Professional, Upgrader, General.
- **Channel** — where it will be published. Options: Blog Post, Instagram Caption, LinkedIn Post, Email Subject Lines.
- **Custom Instructions** — optional. Pin a product model, seasonal angle, or word count ("mention the FitByte Pro", "max 150 words").
- **Generate Content** — runs the full pipeline. Takes roughly 10–15 seconds.

**Left column — User Feedback**
Rate the output with 1–5 stars, leave a comment, and submit. Feedback is saved to `outputs/user_feedback.json`.

**Right column — Output**
- Edit the text directly in the box at any time.
- **Refine with AI** — type an instruction and click Refine. The model applies only that instruction and leaves everything else intact. Word count updates automatically.
- **Approve & Export** — saves the final post as a `.txt` file and downloads it.

---

## Using the CLI

```bash
# Generate a blog post interactively
python app.py --topic "why your resting heart rate matters" --channel blog

# Instagram caption, skip review and auto-save
python app.py --topic "winter training" --channel instagram --auto

# LinkedIn post for health professionals
python app.py --topic "stress and recovery" --channel linkedin --audience health_professional

# Run a uniqueness comparison vs a generic ChatGPT prompt
python app.py --topic "sleep quality" --compare

# Run the pre-defined batch (6 posts across all channels)
python app.py --batch

# Use Anthropic Claude instead of OpenAI
python app.py --topic "overtraining signs" --provider anthropic
```

**Channels:** `blog` | `instagram` | `linkedin` | `email_subject`

**Audiences:** `fitness_enthusiast` | `performance_athlete` | `health_professional` | `upgrader` | `general`

---

## How content stays on-brand

Every prompt injects FitByte's proprietary context before the LLM runs. Nothing is left to the model's defaults.

| What gets injected | Why it matters |
|---|---|
| Brand voice rules | Active voice, specific vocabulary, no hype or exclamation marks |
| Writing rules | Sentence length, em-dash usage, second-person style |
| Real product specs | 14-day battery, dual-frequency GPS, 6-stage sleep tracking — used by name |
| Past content examples | 10 real FitByte posts as few-shot style references |
| Audience insights | Tone and framing adjust per audience segment |
| Market context | Consumer trends and competitor positioning inform the angle |

---

## Content pipeline

Internally, each generation runs five stages:

```
[1] DOCUMENT  — load both knowledge bases from markdown
[2] MONITOR   — LLM scores topic for brand fit and market relevance
[3] BRIEF     — chain-of-thought prompt builds a structured content brief
[4] PUBLISH   — few-shot + contextual prompt generates the final content
[5] ITERATE   — human review: edit, refine, or approve
```

---

## Project structure

```
ai-content-creator/
├── app.py                              # Web app + CLI entry point
├── fitbyte_ai_content_creator.ipynb   # Full walkthrough notebook
├── src/
│   ├── document_processor.py          # Markdown parsing
│   ├── knowledge_base.py              # KB access layer
│   ├── prompt_templates.py            # All prompt templates
│   ├── content_pipeline.py            # Pipeline orchestration
│   └── llm_integration.py            # OpenAI & Anthropic wrapper
├── knowledge_base/
│   ├── primary/                       # Brand guidelines, product specs, examples
│   └── secondary/                     # Market trends, competitor analysis
├── outputs/                           # Generated content, feedback, reports
├── requirements.txt
├── .env.example
└── .gitignore
```

To update the knowledge base, edit the `.md` files in `knowledge_base/`. No reindexing or restarting required.

---

## Notebook

`fitbyte_ai_content_creator.ipynb` walks through the full system step by step:

- loading both knowledge bases and inspecting each component
- previewing the prompt templates (brief, blog, Instagram, LinkedIn)
- running the pipeline stage by stage with visible output at each step
- multi-channel generation from a single topic
- uniqueness demonstration — side-by-side vs a generic ChatGPT prompt
- batch generation across topics and channels
- embedding the Gradio app inline

Run it with `jupyter notebook fitbyte_ai_content_creator.ipynb`.

---

## Reference documents

### Kanban

Task planning and sprint tracking for this project is managed on the Project Trello board. The board tracks backlog, in-progress and completed items across pipeline development, prompt engineering, UI, and evaluation.

Screenshots are located at `kanban/`. Different Porgress stages with the comments as last charackters for End of Day 1, 2 and final

### Human-in-the-Loop Evaluation

`outputs/Human-in-the-Loop Evaluation.md`

Documents how user-driven refinements change AI-generated content across four test cases:

- **Add Metrics** — adding quantified evidence; shifts output from generic to evidence-based
- **Length Control** — "shorten to X words"; model compresses reliably without losing the core message
- **Context Injection** — "tie it to the upcoming Ironman"; strongest single lever for uniqueness
- **Combined Refinement** — metrics + context + word count in one instruction; fastest path to a production-ready post

Covers all four channels and includes cross-channel behavior analysis and key findings.

**Main finding:** Human input is not optional polish — it is what separates a usable post from a generic draft. Context and metrics are the primary drivers of differentiation.

### Uniqueness Comparison

`outputs/uniqueness_comparison.md`

Side-by-side comparison of the FitByte system versus a plain ChatGPT prompt on two topics: sleep quality vs sleep quantity, and rest days and recovery.

Shows the concrete differences across six dimensions: tone, product mention style, writing rules, call-to-action approach, specificity of claims, and brand vocabulary.

**Main finding:** The FitByte system consistently uses named product features ("Sleep Score", "Recovery Score"), avoids generic motivational language, and matches the exact sentence structure of real FitByte posts. The generic output uses vague references, lists, and promotional phrasing that FitByte's brand guidelines explicitly prohibit.

### 
