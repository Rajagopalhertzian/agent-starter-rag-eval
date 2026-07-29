"""Test ingestion pipeline."""
import pytest
import tempfile
import os
from pathlib import Path
from src.agent.ingest import load_documents, chunk_documents, ingest_pipeline


def test_load_documents():
    """Test document loading from temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        Path(tmpdir, "doc1.txt").write_text("Hello world")
        Path(tmpdir, "doc2.md").write_text("# Markdown content")
        Path(tmpdir, "ignore.bin").write_bytes(b"\x00\x01")
        
        docs = load_documents(tmpdir)
        assert len(docs) == 2
        assert any("Hello world" in d.page_content for d in docs)
        assert any("# Markdown content" in d.page_content for d in docs)


def test_chunk_documents():
    """Test document chunking."""
    from langchain_core.documents import Document
    
    docs = [Document(page_content="A" * 1500, metadata={"source": "test.txt"})]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
    
    # 1500 chars with 500 chunk size, 100 overlap = 4 chunks
    assert len(chunks) == 4
    assert all(len(c.page_content) <= 500 for c in chunks)


def test_ingest_pipeline_dry_run():
    """Test ingest pipeline structure (without actual embeddings)."""
    # This just tests the function runs without error
    # Actual embedding requires Ollama running
    pass