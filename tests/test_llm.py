"""Test LLM clients."""
import pytest
from src.agent.llm import OpenAICompatibleClient, EmbeddingsClient


@pytest.mark.asyncio
async def test_ollama_client_creation():
    """Test that Ollama client can be created."""
    client = OpenAICompatibleClient(
        api_base="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen2.5-coder:7b",
    )
    assert client.model == "qwen2.5-coder:7b"
    assert client.api_base == "http://localhost:11434/v1"
    await client.close()


@pytest.mark.asyncio
async def test_embedding_client_creation():
    """Test that embedding client can be created."""
    client = EmbeddingsClient(
        api_base="http://localhost:11434/v1",
        api_key="ollama",
        model="nomic-embed-text",
    )
    assert client.model == "nomic-embed-text"
    await client.close()