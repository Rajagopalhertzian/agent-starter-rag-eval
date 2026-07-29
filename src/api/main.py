"""FastAPI application for RAG agent."""
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent.config import settings
from .agent.graph import rag_graph
from .agent.ingest import ingest_pipeline


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class SourceResponse(BaseModel):
    source: str
    page: int | None = None
    content_preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]
    error: str | None = None


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_created: int
    status: str


class HealthResponse(BaseModel):
    status: str
    model: str
    embedding_model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup/shutdown."""
    # Startup
    print("Starting RAG API...")
    import os
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    os.makedirs(settings.docs_dir, exist_ok=True)
    yield
    # Shutdown
    print("Shutting down RAG API...")


app = FastAPI(
    title="RAG Agent Starter Kit",
    description="Production-ready RAG with local LLMs, evals, and Docker deployment",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model=settings.model_name,
        embedding_model=settings.embedding_model,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system."""
    try:
        result = await rag_graph.ainvoke({"question": request.question})
        return QueryResponse(
            question=request.question,
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            error=result.get("error"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    """Run document ingestion pipeline."""
    try:
        result = ingest_pipeline(settings.docs_dir)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
    )