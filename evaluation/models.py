import atexit
import os
import threading

from openai import OpenAI

_OPENAI_PREFIXES = ("gpt-")
_CLIENTS = {}
_CLIENTS_LOCK = threading.Lock()


def _provider(model: str) -> str:
    override = os.environ.get("MODEL_PROVIDER", "").strip().lower()
    if override:
        return override
    if model.startswith(_OPENAI_PREFIXES):
        return "openai"
    return "vllm"


def _client_config(provider: str) -> tuple[str, str, str]:
    if provider == "openai":
        return (
            "openai",
            os.environ.get("OPENAI_API_KEY", ""),
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    return (
        "vllm",
        os.environ.get("VLLM_API_KEY", "EMPTY"),
        os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
    )


def _get_client(provider: str) -> OpenAI:
    key = _client_config(provider)
    client = _CLIENTS.get(key)
    if client is not None:
        return client

    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = OpenAI(api_key=key[1], base_url=key[2])
            _CLIENTS[key] = client
        return client


@atexit.register
def _close_clients() -> None:
    for client in _CLIENTS.values():
        try:
            client.close()
        except Exception:
            pass


def call_model(
    model: str,
    messages: list,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    client = _get_client(_provider(model))

    if model in {"gpt-5-mini", "gpt-5-nano"}:
        temperature = 1.0

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=120,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
