"""RAG StateGraph using LangGraph."""
from typing import List, Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from .llm import OpenAICompatibleClient, EmbeddingsClient
from .config import settings


class RAGState(TypedDict):
    """State for RAG graph."""
    question: str
    context: List[Document]
    answer: str
    sources: List[Dict[str, Any]]
    error: Optional[str]


class OllamaEmbeddings(Embeddings):
    """LangChain-compatible embeddings wrapper for Ollama."""
    
    def __init__(self, client: EmbeddingsClient):
        self.client = client
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import asyncio
        return asyncio.run(self.client.embed(texts))
    
    def embed_query(self, text: str) -> List[float]:
        import asyncio
        return asyncio.run(self.client.embed([text]))[0]


def get_retriever() -> BaseRetriever:
    """Get configured retriever."""
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
    return vectorstore.as_retriever(search_kwargs={"k": 4})


async def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve relevant documents."""
    try:
        retriever = get_retriever()
        docs = retriever.invoke(state["question"])
        return {**state, "context": docs}
    except Exception as e:
        return {**state, "error": f"Retrieval failed: {e}"}


async def generate_node(state: RAGState) -> RAGState:
    """Generate answer from context."""
    if state.get("error"):
        return state
    
    try:
        llm = OpenAICompatibleClient(
            api_base=settings.model_api_base,
            api_key=settings.model_api_key,
            model=settings.model_name,
            temperature=settings.model_temperature,
        )
        
        # Format context
        context_str = "\n\n".join([
            f"Source {i+1} ({doc.metadata.get('source', 'unknown')}):\n{doc.page_content}"
            for i, doc in enumerate(state["context"])
        ])
        
        messages = [
            {"role": "system", "content": (
                "You are a helpful assistant that answers questions using only the provided context. "
                "Cite sources by their number. If the context doesn't contain the answer, say so."
            )},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {state['question']}"},
        ]
        
        answer = await llm.complete(messages)
        await llm.close()
        
        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", None),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            }
            for doc in state["context"]
        ]
        
        return {**state, "answer": answer, "sources": sources}
    except Exception as e:
        return {**state, "error": f"Generation failed: {e}"}


# Build graph
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_graph = workflow.compile()