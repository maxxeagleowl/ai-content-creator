# Agent Instructions - AI Content Creator

## Objective

Build a working Python MVP for an AI Content Creator.

The system must read markdown documents from two knowledge bases, inject their content into prompt templates, call an LLM API, and generate brand-aligned content that is clearly different from generic ChatGPT output.

Do not build a full RAG system.
Do not use embeddings.
Do not use a vector database.

Focus on a simple working flow:

markdown files â†’ knowledge base context â†’ prompt template â†’ LLM call â†’ generated content

---

## Goal
Build a Python-based AI Content Creator that generates **unique, brand-aligned content** using:
- Primary Knowledge Base (company-specific)
- Secondary Knowledge Base (industry context)

No full RAG. No vector DB. Focus on **markdown â†’ prompt â†’ LLM â†’ output**.

Content pipeline: Implement the workflow: document â†’ monitor â†’ brief â†’ publish â†’ iterate

Document : Ingest and process documents from both knowledge bases (Primary: company-specific, Secondary: industry research)
Monitor : Track and analyze content needs, market trends, and brand alignment
Brief : Generate content briefs based on knowledge base insights
Publish : Create final content ready for publication
Iterate : Refine and improve content based on feedback and updated knowledge

---

## Core Requirements

- LLM API integration (OpenAI or Anthropic)
- Markdown document ingestion
- Two knowledge bases:
  - Primary: brand guidelines, product specs, past content
  - Secondary: market trends, competitors, industry insights
- Prompt templates (min. 2â€“3)
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

## Required Project Structure

ai-content-creator/
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ document_processor.py
â”‚   â”œâ”€â”€ knowledge_base.py
â”‚   â”œâ”€â”€ prompt_templates.py
â”‚   â”œâ”€â”€ content_pipeline.py
â”‚   â”œâ”€â”€ llm_integration.py
â”‚   â””â”€â”€ app.py
â”œâ”€â”€ knowledge_base/
â”‚   â”œâ”€â”€ primary/
â”‚   â”‚   â”œâ”€â”€ brand_guidelines.md
â”‚   â”‚   â”œâ”€â”€ product_specs.md
â”‚   â”‚   â””â”€â”€ past_content.md
â”‚   â””â”€â”€ secondary/
â”‚       â”œâ”€â”€ market_trends.md
â”‚       â””â”€â”€ competitor_analysis.md
â”œâ”€â”€ outputs/
â”œâ”€â”€ config/
â”‚   â””â”€â”€ vscode_agent.json
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ README.md
â”œâ”€â”€ .env.example
â””â”€â”€ .gitignore

---

## Implementation Tasks

### 1. Document Processor

File: src/document_processor.py

Implement:

- load_markdown_file(path: str) -> str
- load_markdown_folder(folder_path: str) -> dict
- clean_markdown(text: str) -> str

Requirements:

- Read .md files only
- Return filename + cleaned content
- Handle missing folders
- Basic error handling

---

### 2. Knowledge Base

File: src/knowledge_base.py

Implement:

- load_primary_knowledge_base() -> dict
- load_secondary_knowledge_base() -> dict
- build_context(primary_docs, secondary_docs) -> str

Requirements:

- Load both folders
- Combine into structured context
- Clearly label primary and secondary

---

### 3. Prompt Templates

File: src/prompt_templates.py

Implement:

- brand_aligned_template(context, topic, content_type)
- industry_context_template(context, topic, content_type)
- hybrid_template(context, topic, content_type)
- executive_linkedin_template(context, topic)
- technical_blog_template(context, topic)

Each template must include:

- role
- task
- injected knowledge base context
- brand voice instruction
- output format
- instruction to avoid generic AI wording

---

### 4. LLM Integration

File: src/llm_integration.py

Implement:

- generate_content(prompt: str) -> str

Requirements:

- Use OPENAI_API_KEY from .env
- Use chat completion
- Return text only
- Handle API errors

---

### 5. Content Pipeline

File: src/content_pipeline.py

Implement:

- run_content_pipeline(topic, content_type, template_type)

Flow:

1. Load knowledge bases
2. Build context
3. Select template
4. Generate content
5. Save to outputs/generated_content.md
6. Return content

---

### 6. Human in the Loop

Add a refinement step after generation.

Implement:

- refine_content(original_content: str, feedback: str) -> str

Flow:

1. Generate draft
2. Save draft
3. Ask user for feedback
4. Send draft + feedback to LLM
5. Save refined version

Files:

outputs/generated_content.md  
outputs/refined_content.md  

---

### 7. Main Entry

File: app.py

Simple CLI:

- ask for topic
- ask for content type
- ask for template
- optionally ask for feedback
- run pipeline

---

### 8. Knowledge Base Content

Create sample markdown files.

Example:

Company: VoltEdge Systems  
Industry: Hydrogen systems  
Product: Modular fuel cell units  
Voice: engineering-driven, precise, confident  

---

### 9. Uniqueness Evidence

Create:

outputs/uniqueness_comparison.md

Structure:

- Topic
- Generic ChatGPT output
- Your system output
- Key differences

Differences must show:

- brand language
- product references
- market context
- non-generic tone

---

### 10. Prompt Iteration Log

Create:

outputs/prompt_iteration_log.md

Document:

- template tested
- issue
- change
- result
- next step

---

### 11. Requirements

requirements.txt:

openai  
python-dotenv  
markdown  

---

### 12. Git Ignore

.gitignore:

.env  
venv/  
__pycache__/  
*.pyc  
.DS_Store  

---

### 13. Environment File

.env.example:

OPENAI_API_KEY=your_api_key_here  

---

## Hivemind Strategies

### 1. Human in the Loop

- Draft + refinement loop
- User feedback required
- Store versions

---

### 2. Style Variation

- Multiple templates
- Different tone and structure
- Avoid repetition patterns

---

### 3. Contextual Placement

- Always use both knowledge bases
- Include company terminology
- Include product + market context
- Output must feel company-specific

---

### 4. Iterative Prompt Engineering

- Test templates
- Improve prompts
- Document iterations
- Track improvements

---

## Coding Style

- Simple functions
- Clear naming
- Minimal complexity
- Basic error handling
- No overengineering

---

## Acceptance Criteria

Project is complete when:

- app.py runs
- markdown is loaded
- context is built
- LLM generates content
- output is saved
- refinement works
- uniqueness comparison exists
- prompt iteration log exists
- README exists
- .env not committed

---

## Important

Do not implement embeddings  
Do not implement vector search  
Do not implement full RAG  

This project proves uniqueness via:

markdown context + prompt engineering

## Output Types

- Blog post
- LinkedIn post
- Marketing copy

---

## Priority

1. Working pipeline
2. Uniqueness
3. Clean structure
4. Documentation

end
