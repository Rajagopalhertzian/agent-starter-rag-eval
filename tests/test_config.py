"""Test the agent configuration."""
import pytest
from src.agent.config import settings


def test_settings_load():
    """Test that settings load correctly."""
    assert settings.model_name == "qwen2.5-coder:7b"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.chroma_collection_name == "rag_documents"
    assert settings.api_port == 8000


def test_settings_api_base():
    """Test API base URLs."""
    assert "11434" in settings.model_api_base
    assert "11434" in settings.embedding_api_base