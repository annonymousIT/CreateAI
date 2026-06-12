"""LLM 抽象と Ollama 実装。

契約は ``docs/interface.md`` §1 を参照。設計の背景は ``docs/design.md`` §4.1、
署名選定の理由は ``docs/tradeoffs.md`` を参照。
"""

from __future__ import annotations

from typing import Protocol

import httpx


class LLMClient(Protocol):
    """LLM への問い合わせを 1 か所に閉じ込めるための薄い Protocol。"""

    def generate(self, prompt: str, *, stop: list[str] | None = None) -> str:
        ...


class OllamaClient:
    """Ollama の HTTP API (``/api/generate``) を叩く同期実装。"""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def generate(self, prompt: str, *, stop: list[str] | None = None) -> str:
        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        if stop:
            # Ollama は options.stop に文字列リストを受け取り、その直前で生成を打ち切る。
            payload["options"] = {"stop": list(stop)}

        response = self._client.post(f"{self._base_url}/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["response"]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
