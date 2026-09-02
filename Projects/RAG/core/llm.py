import os
from typing import Any, Dict, Generator, List, Optional, Tuple
from openai import OpenAI
from .vector_store import VectorStore
from .chunking import DocumentChunk

POPULAR_OPENROUTER_MODELS = [
    {
        "id": "meta-llama/llama-3.3-70b-instruct:free",
        "name": "🎁 Llama 3.3 70B (Meta) - 100% FREE ($0.00)",
        "category": "FREE",
    },
    {
        "id": "google/gemini-2.0-flash-exp:free",
        "name": "🎁 Gemini 2.0 Flash Exp (Google) - 100% FREE ($0.00)",
        "category": "FREE",
    },
    {
        "id": "deepseek/deepseek-r1:free",
        "name": "🎁 DeepSeek R1 Reasoning - 100% FREE ($0.00)",
        "category": "FREE",
    },
    {
        "id": "mistralai/mistral-7b-instruct:free",
        "name": "🎁 Mistral 7B (Mistral) - 100% FREE ($0.00)",
        "category": "FREE",
    },
    {
        "id": "openai/gpt-4o-mini",
        "name": "GPT-4o Mini (OpenAI) - Paid (~$0.0003/query)",
        "category": "Paid / Budget",
    },
    {
        "id": "google/gemini-2.0-flash-001",
        "name": "Gemini 2.0 Flash (Google) - Paid (~$0.0002/query)",
        "category": "Paid / Budget",
    },
    {
        "id": "deepseek/deepseek-chat",
        "name": "DeepSeek V3 (DeepSeek) - Paid (~$0.0003/query)",
        "category": "Paid / Budget",
    },
    {
        "id": "anthropic/claude-3.5-haiku",
        "name": "Claude 3.5 Haiku (Anthropic) - Paid (~$0.001/query)",
        "category": "Paid / Smart",
    },
    {
        "id": "meta-llama/llama-3.3-70b-instruct",
        "name": "Llama 3.3 70B (Meta) - Powerful Open Model",
        "category": "Open Weights",
    },
    {
        "id": "mistralai/mistral-small-24b-instruct-2501",
        "name": "Mistral Small 24B (Mistral) - Precise Reasoning",
        "category": "Open Weights",
    },
    {
        "id": "qwen/qwen-2.5-72b-instruct",
        "name": "Qwen 2.5 72B (Alibaba) - Multilingual & Logical",
        "category": "High Capability",
    },
    {
        "id": "anthropic/claude-3.5-sonnet",
        "name": "Claude 3.5 Sonnet (Anthropic) - Premier Reasoning",
        "category": "Flagship",
    },
    {
        "id": "openai/gpt-4o",
        "name": "GPT-4o (OpenAI) - Flagship Multimodal",
        "category": "Flagship",
    },
]

DEFAULT_SYSTEM_PROMPT = """You are an intelligent, precise RAG (Retrieval-Augmented Generation) assistant.
Your task is to answer the user's questions truthfully and comprehensively using ONLY the provided document context whenever possible.

Guidelines:
1. Synthesize information clearly and concisely.
2. Explicitly cite your sources using bracket notations like [Doc 1], [Doc 2], or [Filename (Page X)] corresponding to the context sections provided.
3. If the provided context does not contain enough information to answer the question, clearly state what information is present and what is missing. Do NOT hallucinate or invent facts outside the provided documents.
4. If the user asks for summaries, comparisons, or analysis across documents, structure your answer using clear bullet points and headings.
"""


class OpenRouterClient:
    """Client for interacting with OpenRouter API using OpenAI SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        site_url: str = "http://localhost:8501",
        app_title: str = "Streamlit RAG App",
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.site_url = site_url
        self.app_title = app_title
        self._client: Optional[OpenAI] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def get_client(self) -> OpenAI:
        if not self.is_configured:
            raise ValueError(
                "OpenRouter API Key is missing. Please provide your OpenRouter API key in the sidebar or in your .env file."
            )
        if self._client is None:
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key.strip(),
                default_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.app_title,
                },
            )
        return self._client

    def test_connection(self, model: str = "openai/gpt-4o-mini") -> Tuple[bool, str]:
        """Tests the OpenRouter API key and model connectivity."""
        try:
            client = self.get_client()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Ping. Respond with 'pong'."}],
                max_tokens=10,
                temperature=0.0,
            )
            reply = response.choices[0].message.content or ""
            return True, f"Connection successful! Model response: {reply.strip()}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> Generator[str, None, None]:
        """Streams response tokens from OpenRouter."""
        client = self.get_client()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
        except Exception as e:
            yield f"\n\n**Error during generation**: `{str(e)}`"

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> str:
        """Non-streaming completion from OpenRouter."""
        client = self.get_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        return response.choices[0].message.content or ""


class RAGPipeline:
    """Orchestrates document context retrieval and LLM prompt construction."""

    def __init__(self, vector_store: VectorStore, llm_client: OpenRouterClient):
        self.vector_store = vector_store
        self.llm_client = llm_client

    def retrieve_context(
        self,
        query: str,
        top_k: int = 4,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Retrieves top relevant chunks from vector store."""
        return self.vector_store.similarity_search(
            query=query,
            top_k=top_k,
            source_filter=source_filter,
        )

    def format_context_prompt(
        self,
        retrieved_items: List[Tuple[DocumentChunk, float]],
    ) -> str:
        """Formats retrieved chunks with citations into structured prompt context."""
        if not retrieved_items:
            return "No relevant context found in the uploaded documents."

        formatted_blocks = []
        for idx, (chunk, score) in enumerate(retrieved_items, start=1):
            source_name = chunk.source
            page_info = f", Page {chunk.page}" if chunk.page and str(chunk.page) != "N/A" else ""
            block = (
                f"--- [Doc {idx}: {source_name}{page_info} | Similarity Score: {score:.2f}] ---\n"
                f"{chunk.content.strip()}\n"
            )
            formatted_blocks.append(block)

        return "\n".join(formatted_blocks)

    def build_messages(
        self,
        query: str,
        retrieved_items: List[Tuple[DocumentChunk, float]],
        chat_history: Optional[List[Dict[str, str]]] = None,
        custom_system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Builds conversation payload with system instructions, context, and message history."""
        system_instruction = custom_system_prompt or DEFAULT_SYSTEM_PROMPT
        context_str = self.format_context_prompt(retrieved_items)

        messages = [
            {"role": "system", "content": system_instruction},
        ]

        # Append recent conversation history (excluding the current query)
        if chat_history:
            # Keep up to 6 recent messages for conversational context
            for msg in chat_history[-6:]:
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

        # User prompt augmented with context
        augmented_user_content = (
            f"Here is the relevant context retrieved from the user's attached documents:\n\n"
            f"{context_str}\n\n"
            f"====================================\n"
            f"User Question: {query}\n\n"
            f"Please answer the question based on the retrieved context above with proper source citations."
        )

        messages.append({"role": "user", "content": augmented_user_content})
        return messages

    def stream_query(
        self,
        query: str,
        model: str = "openai/gpt-4o-mini",
        top_k: int = 4,
        temperature: float = 0.2,
        chat_history: Optional[List[Dict[str, str]]] = None,
        custom_system_prompt: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> Tuple[Generator[str, None, None], List[Tuple[DocumentChunk, float]]]:
        """
        Executes end-to-end RAG query:
        1. Retrieves relevant chunks.
        2. Builds augmented prompt.
        3. Returns token stream and retrieved source chunks.
        """
        retrieved_chunks = self.retrieve_context(
            query=query, top_k=top_k, source_filter=source_filter
        )
        messages = self.build_messages(
            query=query,
            retrieved_items=retrieved_chunks,
            chat_history=chat_history,
            custom_system_prompt=custom_system_prompt,
        )
        stream = self.llm_client.stream_chat(
            messages=messages,
            model=model,
            temperature=temperature,
        )
        return stream, retrieved_chunks
