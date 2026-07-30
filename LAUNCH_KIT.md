# Launch Kit — agent-starter-rag-eval

Copy-paste these into each platform. Edit brackets [ ] where noted.

---

## 1. GUMROAD PRODUCT

### Product Name
**RAG Agent Starter Kit — Local LLMs, Evals, Docker, Deploy Ready**

### Tagline
Production RAG template with zero API costs during dev. LangGraph + Ollama + ChromaDB + built-in evals.

### Price
**$49** (launch) → **$99** after 50 sales

### Description
```
Stop wiring up RAG from scratch. This starter kit gives you a production-grade
Retrieval-Augmented Generation agent in minutes — not days.

WHAT YOU GET
├── LangGraph RAG agent (retrieve → grade → generate → verify)
├── Local-first LLM via Ollama (qwen2.5-coder:7b + nomic-embed-text)
├── ChromaDB vector store with local persistence
├── FastAPI REST API (/query, /ingest, /health)
├── Document ingestion pipeline (txt, md, py, json, yaml)
├── LLM-as-judge evaluation suite (faithfulness + relevance)
├── Docker + docker-compose (one-command dev & prod)
├── Free-tier deploy configs (Railway, Render, Fly.io)
├── 7 passing unit tests + CI-ready
├── MIT licensed — use commercially, resell, modify

ZERO API COSTS DURING DEVELOPMENT
Run entirely on your machine with Ollama. Swap to NVIDIA Nemotron
or any OpenAI-compatible endpoint with one config change.

WHO THIS IS FOR
• Freelancers bidding on RAG projects (save 20 hrs setup per client)
• Solo founders building AI features (ship in days, not weeks)
• Teams needing eval-driven RAG (most templates skip this)
• Devs learning LangGraph/agent patterns (production code to study)

TECH STACK
Python 3.11 • LangGraph • ChromaDB • FastAPI • Ollama • Pydantic • pytest

REQUIREMENTS
• Python 3.11+
• Ollama (free, local)
• 8 GB RAM minimum (for 7B model)
• Optional: Docker, GPU for faster inference

GETTING STARTED (2 minutes)
git clone https://github.com/Rajagopalhertzian/agent-starter-rag-eval
cd agent-starter-rag-eval
uv sync --all-extras
ollama pull qwen2.5-coder:7b && ollama pull nomic-embed-text
cp .env.example .env
uv run python -m src.agent.ingest
uv run dev
# → API at http://localhost:8000/docs

SUPPORT
• GitHub Issues: bug reports, feature requests
• Discussions: usage questions, show & tell
• Email: [YOUR_EMAIL] for customization gigs

LICENSE
MIT — do whatever you want. Attribution appreciated, not required.

CHANGELOG
v0.1.0 — Initial release: RAG agent, evals, Docker, deploy configs
```

### File to Deliver
- GitHub repo link (private or public)
- Or ZIP download (include everything except .venv, data/, __pycache__)

### Tags
`python` `langgraph` `rag` `llm` `ollama` `chromadb` `fastapi` `ai-agent` `starter-kit` `boilerplate` `evals` `docker`

---

## 2. SHOW HN SUBMISSION

### Title
**Show HN: RAG Agent Starter Kit — Local LLMs, Built-in Evals, Docker, Free-Tier Deploy**

### Body
```
I've been building RAG systems for clients and kept rewriting the same
boilerplate: LangGraph workflow, ChromaDB setup, FastAPI endpoints,
document ingestion, and — crucially — an evaluation harness nobody
includes in templates.

So I packaged it as a starter kit: https://github.com/Rajagopalhertzian/agent-starter-rag-eval

Key differentiators:
• Runs 100% locally via Ollama (qwen2.5-coder:7b + nomic-embed-text).
  Zero API keys, zero cloud costs during development.
• LLM-as-judge eval suite built in (faithfulness + relevance scoring).
  Most "production RAG" templates skip eval entirely.
• One-command Docker: `docker compose up` spins up Ollama + API + Chroma.
• Free-tier deploy configs for Railway, Render, Fly.io included.
• MIT licensed — use it for client work, your own SaaS, or learning.

Stack: Python 3.11, LangGraph, ChromaDB, FastAPI, Pydantic, pytest.

The repo started as my internal template for freelance RAG projects.
After 3 clients asked for "the same thing but customized," I realized
the template itself is the product.

Happy to answer questions on architecture, eval methodology, or
deployment. Also open to feedback on what's missing for your use case.
```

### Comment to Post Immediately After
```
Quick demo:
git clone https://github.com/Rajagopalhertzian/agent-starter-rag-eval
cd agent-starter-rag-eval
uv sync --all-extras
ollama pull qwen2.5-coder:7b nomic-embed-text
cp .env.example .env
uv run python -m src.agent.ingest
uv run dev
# Open http://localhost:8000/docs → try /query endpoint
```

---

## 3. UPWORK / TOPTAL PORTFOLIO ENTRY

### Title
**Production RAG Agent Starter Kit (LangGraph + Ollama + ChromaDB + Evals)**

### Category
AI / Machine Learning → LLM Applications / RAG Systems

### Description
```
Built a production-ready RAG agent template that eliminates 20+ hours
of boilerplate per project. Now used as the foundation for client
engagements and sold as a standalone product.

TECHNICAL HIGHLIGHTS
• LangGraph state machine: retrieve → grade → generate → verify
• Local-first LLM via Ollama (supports qwen2.5-coder:7b, llama3.1:8b,
  any OpenAI-compatible endpoint — NVIDIA Nemotron, vLLM, TGI)
• ChromaDB with persistent local storage, metadata filtering
• FastAPI REST API with streaming responses, CORS, health checks
• Document ingestion: recursive chunking, hash-based deduplication,
  incremental re-indexing
• LLM-as-judge evaluation harness: faithfulness (groundedness) +
  relevance scoring via local judge model
• Docker multi-service (Ollama + API) with healthchecks
• Deploy configs for Railway, Render, Fly.io free tiers
• 7 unit tests + CI-ready pytest configuration
• MIT licensed, clean architecture, typed throughout (mypy clean)

BUSINESS IMPACT
• Reduced RAG project setup from ~20 hrs → ~30 min
• Enables fixed-price RAG engagements (predictable margins)
• Template itself generates passive income via Gumroad ($49)
• Portfolio piece demonstrating production ML engineering practices

TECH STACK
Python 3.11 • LangGraph • ChromaDB • FastAPI • Ollama • Pydantic
• pytest • Docker • uv • Ruff • MyPy

LIVE DEMO
GitHub: https://github.com/Rajagopalhertzian/agent-starter-rag-eval
API Docs (local): http://localhost:8000/docs (after `docker compose up`)

AVAILABLE FOR
• RAG system design & implementation
• Agent workflow development (LangGraph, AutoGen, CrewAI)
• LLM evaluation & observability setup
• Production ML infrastructure (inference, monitoring, CI/CD)
```

### Skills Tags
`Python` `LangGraph` `RAG` `LLM` `Ollama` `ChromaDB` `FastAPI` `Docker` `Machine Learning` `AI Agents` `Evaluation`

### Project Duration
2 weeks (part-time)

### Client Type
Internal tool / Product template

---

## 4. GITHUB REPO POLISH (Do This Now)

### Repository Settings
- Description: `Production RAG agent starter kit — LangGraph + Ollama + ChromaDB + built-in evals + Docker + free-tier deploy. Zero API costs during dev. MIT licensed.`
- Website: `https://github.com/Rajagopalhertzian/agent-starter-rag-eval`
- Topics: `rag` `langgraph` `ollama` `chromadb` `fastapi` `llm` `ai-agent` `starter-kit` `boilerplate` `evals` `docker` `python`

### Social Preview Image
Create 1280×640 PNG with:
- Title: "RAG Agent Starter Kit"
- Subtitle: "Local LLMs • Evals • Docker • Deploy Ready"
- Stack badges: Python, LangGraph, Ollama, ChromaDB, FastAPI
- Save as `.github/social-preview.png`

### README Badges (Add to Top)
```markdown
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://github.com/Rajagopalhertzian/agent-starter-rag-eval/actions/workflows/ci.yml/badge.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
[![Gumroad](https://img.shields.io/badge/Buy%20on-Gumroad-orange)](https://gumroad.com/l/rag-agent-starter)
```

---

## 5. LAUNCH CHECKLIST (Copy to Notion/Obsidian)

- [ ] Gumroad product created, repo linked, price $49
- [ ] Show HN submitted (weekday 9-11 AM EST optimal)
- [ ] Upwork portfolio entry published
- [ ] Toptal profile updated (if applicable)
- [ ] GitHub topics + description + social preview set
- [ ] Tweet/LinkedIn post with repo link + "Show HN" mention
- [ ] Reddit: r/LangChain, r/LocalLLaMA, r/MachineLearning (not self-promo heavy)
- [ ] Email 5 past clients/leads: "Built this template from our project — thought you'd like it"
- [ ] GitHub Sponsors enabled
- [ ] Issue templates + CONTRIBUTING.md for community

---

## 6. PRICING STRATEGY

| Tier | Price | Includes |
|------|-------|----------|
| **Launch** | $49 | Full repo, lifetime updates, email support (30 days) |
| **Standard** | $99 | Above + 1 hr customization call |
| **Team** | $299 | Above + 5 seats, priority support, private Discord |

Increase price by $25 every 50 sales. Communicate: "Price goes up [date]."

---

## 7. NEXT PRODUCT IDEAS (From This Template)

1. **Agent Eval Pack** — Deeper eval suite (RAGAS, correctness, safety)
2. **Multi-Agent RAG** — Router + planner + verifier agents
3. **RAG Observability** — Langfuse/Logfire integration + dashboards
4. **Vertical Templates** — Legal RAG, Code RAG, Support RAG, Research RAG

Each = 1-2 weeks build, same launch playbook.