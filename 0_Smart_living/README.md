# 🌿 GreenPulse AI Content Creator

**Smart Living, Green Future**

An AI-powered content creation system that generates brand-aligned marketing content for GreenPulse, a fictional eco-friendly smart home company. The system uses a two-layer knowledge base and advanced prompt engineering to produce unique, on-brand content that avoids generic AI patterns.

---

## 📌 Project Overview

This project demonstrates how to integrate LLM APIs with structured knowledge bases to create a dynamic content generation pipeline. Instead of producing generic AI content, the system leverages company-specific brand guidelines, product specifications, and industry research to generate authentic, brand-aligned content.

### The Problem
Generic AI content suffers from homogenization — everyone gets similar outputs regardless of context. Audiences now recognize and distrust generic AI-generated content.

### The Solution
GreenPulse AI Content Creator solves this by:
- Injecting real brand context into every prompt
- Using advanced prompt templates tailored to different content types
- Implementing a 5-stage content pipeline with built-in refinement
- Providing automatic fallback between LLM providers

---

## 🏗️ Project Structure
ai-content-creator/
├── main.ipynb                       # Main application (Jupyter Notebook)
├── knowledge_base/
│ ├── primary/                       # Company-specific documents
│ │ ├── brand_guidelines.md
│ │ ├── product_specs.md
│ │ └── past_content/
│ │ └── successful_blog_example.md
│ └── secondary/                     # Industry research documents
│ ├── market_trends.md
│ └── competitor_analysis.md
├── config/
│ └── vscode_agent.json              # VSCode agent configuration
├── requirements.txt
├── README.md
└── .env # API keys (not committed)

---

## 🔧 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Cohere API key
- VSCode with Jupyter extension

### Step 1: Clone the Repository
```bash
git clone https://github.com/Adetunji/ai-content-creator.git
cd ai-content-creator
Step 2: Create Virtual Environment
bash
python -m venv venv
Activate it:

Windows: venv\Scripts\activate
Mac/Linux: source venv/bin/activate
Step 3: Install Dependencies
bash
pip install -r requirements.txt
Step 4: Set Up Environment Variables

Create a .env file in the project root:

OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
⚠️ Never commit your .env file to GitHub!

Step 5: Run the Notebook
Open main.ipynb in VSCode
Select the Python kernel from your virtual environment
Run all cells from top to bottom
📦 Dependencies
Package	Purpose
openai	Primary LLM API integration
cohere	Fallback LLM API integration
markdown	Markdown document parsing
python-dotenv	Environment variable management
mistune	Additional markdown processing
🌿 About GreenPulse (Fictional Company)
Detail	Description
Company	GreenPulse
Industry	Eco-friendly smart home products
Products	EcoTherm Smart Thermostat, SolarSense Monitor, PurePlug Smart Outlet
Brand-Voice	Friendly, optimistic, practical, approachable
Target-Audience	Eco-conscious homeowners aged 25-45
Tagline	Smart Living, Green Future
📚 Two-Layer Knowledge Base
Primary Knowledge Base (Company-Specific)
Brand_Guidelines — Voice, tone, messaging principles, words to use/avoid
Product_Specifications — Features, pricing, and benefits for all products
Past_Successful_Content — Examples of high-performing content with analysis
Secondary Research Layer (Industry Context)
Market_Trends — Consumer preferences, industry growth, content trends
Competitor_Analysis — Competitor strengths, weaknesses, and opportunities

🔄 Content Pipeline
The system implements a 5-stage content creation workflow:

Stage	Description
1. Document	Ingest and process markdown files from both knowledge bases
2. Monitor	Analyze content needs, market trends, and brand alignment
3. Brief	Generate content brief with topic, audience, goals, and guidelines
4. Publish	Generate final content using LLM with full knowledge base context
5. Iterate	Refine and improve content based on feedback
✍️ Content Types Supported
📝 Blog Posts
📱 Social Media Posts (Instagram, Twitter, LinkedIn)
📧 Email Newsletters
🏷️ Product Descriptions
🤖 LLM Integration
Primary Provider: OpenAI (GPT-4o-mini)
Fallback Provider: Cohere (Command)
Automatic failover if OpenAI is unavailable
Configurable temperature and model settings
🎯 Uniqueness Strategies
This project avoids generic AI content through:

Knowledge Base Context — Every prompt includes company-specific and industry context
Advanced Prompt Templates — Templates enforce brand voice, tone, and messaging rules
Iterative Refinement — Content passes through a refinement stage to remove generic patterns
Style Variation — Different templates for different content types and platforms
Human in the Loop — Pipeline supports feedback-based iteration
Uniqueness Evidence
The notebook includes a side-by-side comparison showing:

Generic AI output (no context) vs. Brand-aligned output (with full knowledge base)
Clear differences in tone, specificity, product mentions, and audience targeting
⚙️ VSCode Agent Configuration
The project includes a VSCode agent configuration file at config/vscode_agent.json with:

Recommended extensions
Python environment settings
Development workflow steps
📋 Kanban Board
This project was managed using a Trello Kanban board with:

Backlog → In Progress (WIP Limit: 3) → Review/Testing → Done
Granular task cards for each feature
Definition of Done documented on the board
Day 1 and Day 2 screenshots showing progression
🚀 How to Use
Run all cells in main.ipynb
The notebook will:
Create knowledge base folders and documents
Load and process all markdown files
Initialize LLM connections
Run the full content pipeline
Generate multiple content types
Show uniqueness comparison
👤 Author
Name: [PROJECT GROÜP 8]
Project: AI Content Creator — Project 2
Date: [05-05-2026]
📜 License
This project is for educational purposes only.