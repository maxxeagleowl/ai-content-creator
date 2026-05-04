# AI Content Creator - Agent Context

## Goal
Build a Python-based AI Content Creator that generates **unique, brand-aligned content** using:
- Primary Knowledge Base (company-specific)
- Secondary Knowledge Base (industry context)

No full RAG. No vector DB. Focus on **markdown → prompt → LLM → output**.

---

## Core Requirements

- LLM API integration (OpenAI or Anthropic)
- Markdown document ingestion
- Two knowledge bases:
  - Primary: brand guidelines, product specs, past content
  - Secondary: market trends, competitors, industry insights
- Prompt templates (min. 2–3)
- Content generation pipeline
- Demonstrate **uniqueness vs generic ChatGPT**

---

## MVP Scope

Focus ONLY on:
1. Read markdown files
2. Extract relevant context
3. Inject into prompt
4. Generate content via LLM

Optional later:
- monitor / brief / iterate stages
- formatting / export

---

## Project Structure
src/
document_processor.py
knowledge_base.py
prompt_templates.py
content_pipeline.py
llm_integration.py
main.py

knowledge_base/
primary/
secondary/

templates/
config/
---
## Key Modules

### document_processor.py
- Load markdown files
- Clean + structure text
- Return usable chunks

### knowledge_base.py
- Load primary + secondary docs
- Provide context for prompts

### prompt_templates.py
- Template 1: Brand-focused
- Template 2: Industry-focused
- Template 3: Hybrid

### llm_integration.py
- API call wrapper
- Input: prompt
- Output: generated text

### content_pipeline.py
- Combine:
  - documents
  - prompt templates
  - LLM call

### main.py
- Entry point
- Run full flow

---

## Prompt Strategy

Always include:
- Brand voice (Primary KB)
- Market context (Secondary KB)
- Clear task instruction
- Style constraint

Avoid generic phrasing.

---

## Uniqueness Strategy

- Inject real KB content into prompts
- Use specific terminology
- Reference products / strategy
- Avoid generic tone

Test:
- Generate same content with plain ChatGPT
- Compare outputs side by side

---

## Output Types

- Blog post
- LinkedIn post
- Marketing copy

---

## Constraints

- No embeddings required
- No vector DB required
- Keep implementation simple
- Focus on working pipeline

---

## Success Criteria

- Content uses both KBs
- Output is clearly non-generic
- Pipeline runs end-to-end
- Code is modular and clean

---

## Dev Notes

- Use `.env` for API keys
- Do not commit secrets
- Keep functions small and reusable
- Print intermediate steps for debugging

---

## Workflow

1. Load documents
2. Build context
3. Select template
4. Generate content
5. Compare with baseline

---

## Priority

1. Working pipeline
2. Uniqueness
3. Clean structure
4. Documentation