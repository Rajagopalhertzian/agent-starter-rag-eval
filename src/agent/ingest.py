"""Document ingestion pipeline."""
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.agent.config import settings
from src.agent.llm import EmbeddingsClient


class OllamaEmbeddings:
    """LangChain-compatible embeddings wrapper for Ollama."""
    
    def __init__(self, client: EmbeddingsClient):
        self.client = client
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        return asyncio.run(self.client.embed(texts))
    
    def embed_query(self, text: str) -> list[float]:
        import asyncio
        return asyncio.run(self.client.embed([text]))[0]


def load_documents(data_dir: str = "./data/documents") -> list[Document]:
    """Load documents from directory."""
    docs = []
    path = Path(data_dir)
    
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        # Create sample doc
        sample = path / "sample.txt"
        sample.write_text(
            "This is a sample document for the RAG agent starter kit. "
            "It contains information about the project architecture, "
            "components, and usage. The project uses LangGraph for orchestration, "
            "ChromaDB for vector storage, and Ollama for local LLM inference."
        )
        return [Document(page_content=sample.read_text(), metadata={"source": "sample.txt"})]
    
    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix in [".txt", ".md", ".py", ".json", ".yaml", ".yml"]:
            try:
                content = file_path.read_text(encoding="utf-8")
                docs.append(Document(
                    page_content=content,
                    metadata={"source": str(file_path.relative_to(path))}
                ))
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
    
    return docs


def chunk_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest_pipeline(data_dir: str = "./data/documents") -> dict[str, Any]:
    """Run full ingestion pipeline."""
    print("Loading documents...")
    docs = load_documents(data_dir)
    print(f"Loaded {len(docs)} documents")
    
    print("Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")
    
    print("Generating embeddings and storing...")
    embedding_client = EmbeddingsClient(
        api_base=settings.embedding_api_base,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
    )
    embeddings = OllamaEmbeddings(embedding_client)
    
    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        persist_directory=settings.chroma_persist_dir,
        embedding_function=embeddings,
    )
    
    vectorstore.add_documents(chunks)
    
    return {
        "documents_loaded": len(docs),
        "chunks_created": len(chunks),
        "status": "success",
    }


if __name__ == "__main__":
    result = ingest_pipeline()
    print(result)