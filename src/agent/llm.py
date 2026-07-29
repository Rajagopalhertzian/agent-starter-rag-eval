"""OpenAI-compatible client for local Ollama and NVIDIA APIs."""
import httpx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    choices: List[ChatCompletionChoice]
    model: str


class OpenAICompatibleClient:
    """Client for OpenAI-compatible APIs (Ollama, NVIDIA, etc.)."""
    
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        timeout: float = 60.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        return self._client
    
    async def complete(self, messages: List[Dict[str, str]]) -> str:
        """Complete chat and return content."""
        request = ChatCompletionRequest(
            model=self.model,
            messages=[Message(**m) for m in messages],
            temperature=self.temperature,
        )
        
        response = await self.client.post(
            "/chat/completions",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        
        result = ChatCompletionResponse(**response.json())
        return result.choices[0].message.content
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


class EmbeddingsClient:
    """Client for embeddings API."""
    
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        return self._client
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        response = await self.client.post(
            "/embeddings",
            json={"model": self.model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None