# RAG Agent Starter Kit

Production-ready RAG (Retrieval-Augmented Generation) agent template with:
- **Local LLMs** via Ollama (no API costs during development)
- **LangGraph** for agent orchestration
- **ChromaDB** for vector storage
- **FastAPI** for REST API
- **Built-in evals** with LLM-as-judge
- **Docker** for one-command deployment
- **Free-tier deploy configs** for Railway/Render/Fly.io

## Quickstart

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- [Docker](https://docker.com) (optional, for containerized deployment)

### Local Development (No Docker)

```bash
# 1. Clone and enter
git clone https://github.com/Rajagopalhertzian/agent-starter-rag-eval.git
cd agent-starter-rag-eval

# 2. Install dependencies
uv sync --all-extras

# 3. Start Ollama and pull models
ollama serve &
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text

# 4. Configure environment
cp .env.example .env

# 5. Add documents to ./data/docs/
# (supports .txt, .md, .py, .json, .yaml, .yml)

# 6. Ingest documents
uv run python -m src.agent.ingest

# 7. Start API server
uv run dev
# → API at http://localhost:8000
# → Docs at http://localhost:8000/docs
```

### Docker (One Command)

```bash
# Build and start everything
docker compose up -d

# Pull models (first run only)
docker compose exec ollama ollama pull qwen2.5-coder:7b
docker compose exec ollama ollama pull nomic-embed-text

# Ingest documents
docker compose exec api uv run python -m src.agent.ingest

# API at http://localhost:8000
```

## Project Structure

```
agent-starter-rag-eval/
├── src/
│   ├── agent/           # Core RAG agent
│   │   ├── config.py    # Settings management
│   │   ├── llm.py       # LLM/embedding clients
│   │   ├── graph.py     # LangGraph RAG workflow
│   │   └── ingest.py    # Document ingestion pipeline
│   ├── api/
│   │   └── main.py      # FastAPI application
│   └── eval/
│       └── runner.py    # LLM-as-judge evaluation
├── tests/               # Unit + integration + eval tests
├── config/
│   └── models.yaml      # Model provider configs
├── deploy/              # Free-tier deploy configs
├── data/
│   ├── docs/            # Your documents go here
│   └── chroma/          # Vector DB (auto-created)
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── pyproject.toml
```

## API Usage

```bash
# Query the RAG system
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this project?"}'

# Response
{
  "question": "What is this project?",
  "answer": "A RAG agent starter kit...",
  "sources": [
    {"source": "README.md", "content_preview": "RAG Agent Starter Kit..."}
  ]
}

# Trigger re-ingestion
curl -X POST http://localhost:8000/ingest

# Health check
curl http://localhost:8000/health
```

## Evaluation

Run the built-in eval suite (uses local Ollama as judge):

```bash
uv run pytest tests/ -v
```

Or run evals directly:

```bash
uv run python -m src.eval.runner
```

The eval suite tests:
- **Faithfulness**: Does the answer stay grounded in retrieved context?
- **Relevance**: Does the answer address the question?

## Configuration

### Model Providers

Edit `config/models.yaml` to switch between:
- **Ollama** (local, free) — default
- **NVIDIA API** (cloud, free tier available)

```yaml
models:
  ollama:
    provider: "ollama"
    model: "qwen2.5-coder:7b"
    api_base: "http://localhost:11434/v1"
    api_key: "ollama"
  nvidia:
    provider: "nvidia"
    model: "nvidia/nemotron-3-ultra"
    api_base: "https://integrate.api.nvidia.com/v1"
    api_key_env: "NVIDIA_API_KEY"

default: "ollama"
```

### Environment Variables

See `.env.example` for all options. Key ones:
- `MODEL_NAME` — Ollama model (default: `qwen2.5-coder:7b`)
- `EMBEDDING_MODEL` — Embedding model (default: `nomic-embed-text`)
- `CHROMA_PERSIST_DIR` — Vector DB location
- `DOCS_DIR` — Document source directory

## Deployment (Free Tiers)

### Railway
```bash
# 1. Push to GitHub
# 2. Connect repo at railway.app
# 3. Add environment variables from .env.example
# 4. Deploy → auto-detects Dockerfile
```

### Render
```bash
# 1. Push to GitHub
# 2. Create Web Service at render.com
# 3. Select Docker runtime
# 4. Add environment variables
```

### Fly.io
```bash
fly launch --dockerfile Dockerfile
fly secrets set MODEL_API_KEY=ollama EMBEDDING_API_KEY=ollama
fly deploy
```

## Adding Documents

Place files in `./data/docs/` (or `DOCS_DIR`):
```
data/docs/
├── company-policy.txt
├── api-docs.md
├── faq.json
└── guides/
    └── getting-started.md
```

Supported formats: `.txt`, `.md`, `.py`, `.json`, `.yaml`, `.yml`

Then run ingestion:
```bash
uv run python -m src.agent.ingest
```

## Customization

### Change the RAG Prompt
Edit `src/agent/graph.py` → `generate_node` system message.

### Add Evaluation Cases
Add to `tests/fixtures/eval_cases.json`:
```json
{
  "question": "Your question",
  "expected_answer": "Expected answer",
  "expected_sources": ["source1.md"]
}
```

### Add API Endpoints
Edit `src/api/main.py` — standard FastAPI patterns.

## Why This Template?

| Feature | Most Templates | This Template |
|---------|----------------|---------------|
| Local LLM support | ❌ | ✅ Ollama first-class |
| Evaluation suite | ❌ | ✅ LLM-as-judge built-in |
| Vector DB | Chroma only | ✅ Chroma (swappable) |
| Docker | Basic | ✅ Multi-service with healthchecks |
| Free deploy | Manual | ✅ Railway/Render/Fly configs |
| Observability | ❌ | Ready for Langfuse/Logfire |

## License

MIT — Use freely for commercial or personal projects.

## Support

- Issues: [GitHub Issues](https://github.com/Rajagopalhertzian/agent-starter-rag-eval/issues)
- Discussions: [GitHub Discussions](https://github.com/Rajagopalhertzian/agent-starter-rag-eval/discussions)

---

**Built for developers who want to ship RAG applications, not boilerplate.**