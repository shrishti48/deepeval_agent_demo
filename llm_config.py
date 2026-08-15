import hashlib
import itertools
import os
import re
import threading
from typing import Any, Iterable, Sequence

from deepeval.models import OpenRouterModel
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from openai import APIStatusError, AsyncOpenAI, OpenAI, RateLimitError
from tenacity import RetryError

load_dotenv()

DEFAULT_OPENROUTER_MODELS = [
    "openai/gpt-oss-20b:free",
    "google/gemma-4-26b-a4b-it:free",
    "cohere/north-mini-code:free",
    "google/gemma-4-31b-it:free",
    "openrouter/free",
]
DEFAULT_OPENROUTER_JUDGE_MODELS = [
    "openai/gpt-oss-20b:free",
    "google/gemma-4-26b-a4b-it:free",
    "cohere/north-mini-code:free",
    "google/gemma-4-31b-it:free",
    "openrouter/free",
]
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_model_counter = itertools.count()
_model_counter_lock = threading.Lock()


def get_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY. Add it to your environment or .env file."
        )
    return api_key


def get_openrouter_base_url() -> str:
    return os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)


def get_openrouter_model_names() -> list[str]:
    models_env = os.getenv("OPENROUTER_MODELS")
    if models_env:
        models = [model.strip() for model in models_env.split(",") if model.strip()]
        if models:
            return models

    model = os.getenv("OPENROUTER_MODEL")
    if model:
        return [model]

    return DEFAULT_OPENROUTER_MODELS[:]


def get_openrouter_judge_model_names() -> list[str]:
    models_env = os.getenv("OPENROUTER_JUDGE_MODELS")
    if models_env:
        models = [model.strip() for model in models_env.split(",") if model.strip()]
        if models:
            return models

    return DEFAULT_OPENROUTER_JUDGE_MODELS[:]


def get_openrouter_model_name() -> str:
    return get_openrouter_model_names()[0]


def _next_model_order(models: Sequence[str]) -> list[str]:
    with _model_counter_lock:
        start = next(_model_counter) % len(models)
    return [models[(start + offset) % len(models)] for offset in range(len(models))]


def _unwrap_retry_error(exc: Exception) -> Exception:
    current = exc
    while isinstance(current, RetryError):
        outcome = getattr(current, "last_attempt", None)
        if outcome is None:
            break
        try:
            outcome.result()
        except Exception as inner_exc:  # pragma: no cover - depends on remote provider
            current = inner_exc
            continue
        break
    return current


def _should_try_next_model(exc: Exception) -> bool:
    exc = _unwrap_retry_error(exc)
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError):
        if exc.status_code in {404, 408, 409, 429, 500, 502, 503, 504}:
            return True
        message = str(exc).lower()
        return "unavailable for free" in message or "temporarily rate-limited" in message
    status_code = getattr(exc, "status_code", None)
    if status_code in {404, 408, 409, 429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return "unavailable for free" in message or "temporarily rate-limited" in message


def _build_chat_openai(model: str, *, temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=get_openrouter_api_key(),
        base_url=get_openrouter_base_url(),
        temperature=temperature,
        use_responses_api=False,
    )


def create_openrouter_chat_completion(**kwargs: Any):
    last_error = None
    for model in _next_model_order(get_openrouter_model_names()):
        client = OpenAI(
            api_key=get_openrouter_api_key(),
            base_url=get_openrouter_base_url(),
        )
        try:
            return client.chat.completions.create(model=model, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on remote provider
            last_error = exc
            if not _should_try_next_model(exc):
                raise
    raise last_error


class RotatingOpenRouterChatModel(BaseChatModel):
    model_names: list[str]
    temperature: float = 0
    bound_tools: tuple[Any, ...] = ()
    tool_kwargs: dict[str, Any] = {}

    @property
    def _llm_type(self) -> str:
        return "rotating-openrouter"

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        tool_kwargs = dict(kwargs)
        if tool_choice is not None:
            tool_kwargs["tool_choice"] = tool_choice
        return RotatingOpenRouterChatModel(
            model_names=self.model_names,
            temperature=self.temperature,
            bound_tools=tuple(tools),
            tool_kwargs=tool_kwargs,
        )

    def _build_attempt(self, model: str):
        attempt = _build_chat_openai(model, temperature=self.temperature)
        if self.bound_tools:
            attempt = attempt.bind_tools(self.bound_tools, **self.tool_kwargs)
        return attempt

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        last_error = None
        for model in _next_model_order(self.model_names):
            try:
                message = self._build_attempt(model).invoke(messages, stop=stop, **kwargs)
                return ChatResult(generations=[ChatGeneration(message=message)])
            except Exception as exc:  # pragma: no cover - depends on remote provider
                last_error = exc
                if not _should_try_next_model(exc):
                    raise
        raise last_error

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        last_error = None
        for model in _next_model_order(self.model_names):
            try:
                message = await self._build_attempt(model).ainvoke(
                    messages, stop=stop, **kwargs
                )
                return ChatResult(generations=[ChatGeneration(message=message)])
            except Exception as exc:  # pragma: no cover - depends on remote provider
                last_error = exc
                if not _should_try_next_model(exc):
                    raise
        raise last_error


class RotatingOpenRouterDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_names: Sequence[str] | None = None):
        self.model_names = list(model_names or get_openrouter_judge_model_names())
        self.temperature = 0.0
        super().__init__(self.model_names[0])

    def load_model(self, *args, **kwargs):
        return self

    def generate(self, prompt: str, schema=None):
        last_error = None
        for model in _next_model_order(self.model_names):
            try:
                return OpenRouterModel(
                    model=model,
                    api_key=get_openrouter_api_key(),
                    base_url=get_openrouter_base_url(),
                    temperature=self.temperature,
                ).generate(prompt, schema=schema)
            except Exception as exc:  # pragma: no cover - depends on remote provider
                last_error = exc
                if not _should_try_next_model(exc):
                    raise
        raise last_error

    async def a_generate(self, prompt: str, schema=None):
        last_error = None
        for model in _next_model_order(self.model_names):
            try:
                return await OpenRouterModel(
                    model=model,
                    api_key=get_openrouter_api_key(),
                    base_url=get_openrouter_base_url(),
                    temperature=self.temperature,
                ).a_generate(prompt, schema=schema)
            except Exception as exc:  # pragma: no cover - depends on remote provider
                last_error = exc
                if not _should_try_next_model(exc):
                    raise
        raise last_error

    def get_model_name(self, *args, **kwargs) -> str:
        return f"{self.model_names[0]}+fallbacks (OpenRouter)"


def build_openai_client() -> OpenAI:
    return OpenAI(
        api_key=get_openrouter_api_key(),
        base_url=get_openrouter_base_url(),
    )


def build_async_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=get_openrouter_api_key(),
        base_url=get_openrouter_base_url(),
    )


def build_chat_model(*, temperature: float = 0) -> RotatingOpenRouterChatModel:
    return RotatingOpenRouterChatModel(
        model_names=get_openrouter_model_names(),
        temperature=temperature,
    )


def build_deepeval_model() -> DeepEvalBaseLLM:
    return RotatingOpenRouterDeepEvalModel()


class LocalHashEmbeddings(Embeddings):
    """Deterministic local embeddings to keep the demo free-key-only."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return vector

    @staticmethod
    def _tokenize(text: str) -> Iterable[str]:
        return re.findall(r"[a-z0-9]+", text.lower())
