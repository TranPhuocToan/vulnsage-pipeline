"""Strict Ollama transport for the canonical runner.

Importing this module never starts Ollama, downloads a model, or sends a
request.  Network I/O happens only when an explicit method is called.  Tests
inject a fake HTTP session, so the integration contract is verifiable offline.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

import requests


class OllamaTransportError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaConfig:
    model: str = "qwen2.5-coder:7b"
    base_url: str = "http://127.0.0.1:11434"
    timeout_sec: float = 600.0
    temperature: float = 0.0
    num_ctx: int = 32768
    num_predict: int = 2048
    seed: int = 25550029

    def __post_init__(self) -> None:
        if self.model != "qwen2.5-coder:7b":
            raise ValueError("Phase 3D model must be exactly qwen2.5-coder:7b")
        if self.base_url.rstrip("/") not in {
            "http://127.0.0.1:11434", "http://localhost:11434"
        }:
            raise ValueError("Ollama base_url must be the local Kaggle service")
        if self.timeout_sec <= 0 or self.num_ctx <= 0 or self.num_predict <= 0:
            raise ValueError("timeout and context/output limits must be positive")
        if self.temperature != 0.0:
            raise ValueError("temperature must remain 0.0 for the experiment")

    def run_parameters(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "seed": self.seed,
            "stream": False,
        }


class OllamaGenerateTransport:
    """Callable adapter compatible with ``StructuredLLMClient``."""

    def __init__(
        self,
        config: OllamaConfig,
        *,
        session: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.clock = clock
        self.call_log: list[dict[str, Any]] = []

    @property
    def base_url(self) -> str:
        return self.config.base_url.rstrip("/")

    def _json_response(self, response: Any, endpoint: str) -> Mapping[str, Any]:
        status = getattr(response, "status_code", None)
        if status != 200:
            body = str(getattr(response, "text", ""))[:500]
            raise OllamaTransportError(f"OLLAMA_HTTP_ERROR:{endpoint}:{status}:{body}")
        try:
            payload = response.json()
        except Exception as exc:
            raise OllamaTransportError(f"OLLAMA_INVALID_JSON:{endpoint}:{exc}") from exc
        if not isinstance(payload, Mapping):
            raise OllamaTransportError(f"OLLAMA_INVALID_PAYLOAD:{endpoint}")
        return payload

    def probe(self) -> dict[str, Any]:
        """Return exact-model availability and immutable model metadata."""
        try:
            tags_response = self.session.get(
                f"{self.base_url}/api/tags", timeout=self.config.timeout_sec
            )
        except Exception as exc:
            raise OllamaTransportError(f"OLLAMA_UNREACHABLE:{exc}") from exc
        tags = self._json_response(tags_response, "tags")
        models = tags.get("models")
        if not isinstance(models, list):
            raise OllamaTransportError("OLLAMA_TAGS_MODELS_INVALID")
        exact = next(
            (item for item in models if isinstance(item, Mapping) and item.get("name") == self.config.model),
            None,
        )
        if exact is None:
            return {
                "available": False,
                "requested_model": self.config.model,
                "available_models": sorted(
                    str(item.get("name")) for item in models
                    if isinstance(item, Mapping) and item.get("name")
                ),
            }
        try:
            show_response = self.session.post(
                f"{self.base_url}/api/show",
                json={"model": self.config.model},
                timeout=self.config.timeout_sec,
            )
        except Exception as exc:
            raise OllamaTransportError(f"OLLAMA_SHOW_FAILED:{exc}") from exc
        show = self._json_response(show_response, "show")
        metadata = {
            "available": True,
            "requested_model": self.config.model,
            "digest": exact.get("digest"),
            "size": exact.get("size"),
            "modified_at": exact.get("modified_at"),
            "details": exact.get("details"),
            "model_info": show.get("model_info"),
            "parameters": show.get("parameters"),
            "template_sha256": hashlib.sha256(
                str(show.get("template", "")).encode("utf-8")
            ).hexdigest(),
        }
        if not isinstance(metadata["digest"], str) or not metadata["digest"]:
            raise OllamaTransportError("OLLAMA_MODEL_DIGEST_MISSING")
        return metadata

    def __call__(self, prompt: str) -> tuple[str, dict[str, int | None]]:
        if not isinstance(prompt, str) or not prompt:
            raise OllamaTransportError("OLLAMA_PROMPT_EMPTY")
        request = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": self.config.num_ctx,
                "num_predict": self.config.num_predict,
                "seed": self.config.seed,
            },
        }
        started = self.clock()
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=request,
                timeout=self.config.timeout_sec,
            )
            payload = self._json_response(response, "generate")
        except Exception as exc:
            self.call_log.append({
                "call": len(self.call_log) + 1,
                "status": "ERROR",
                "duration_sec": max(0.0, self.clock() - started),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            raise
        text = payload.get("response")
        if not isinstance(text, str) or not text:
            raise OllamaTransportError("OLLAMA_RESPONSE_TEXT_MISSING")
        input_tokens = payload.get("prompt_eval_count")
        output_tokens = payload.get("eval_count")
        if type(input_tokens) is not int:
            input_tokens = None
        if type(output_tokens) is not int:
            output_tokens = None
        total_tokens = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None else None
        )
        self.call_log.append({
            "call": len(self.call_log) + 1,
            "status": "OK",
            "duration_sec": max(0.0, self.clock() - started),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "done": payload.get("done"),
            "done_reason": payload.get("done_reason"),
            "total_duration_ns": payload.get("total_duration"),
            "load_duration_ns": payload.get("load_duration"),
            "prompt_eval_duration_ns": payload.get("prompt_eval_duration"),
            "eval_duration_ns": payload.get("eval_duration"),
        })
        return text, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
