"""
Local transformers + PEFT (LoRA) provider for SafeClaw.

This provider runs a HuggingFace causal-LM locally — optionally with a
LoRA adapter loaded on top of the base model — so users can serve a
fine-tuned coding assistant without depending on an external server.

The heavy dependencies (``torch``, ``transformers``, ``peft``,
``huggingface_hub``) are imported lazily so that the rest of SafeClaw
keeps working when the optional ``[finetune]`` extra is not installed.

Configure it from ``config/config.yaml``::

    ai_providers:
      - label: safeclaw-coder
        provider: transformers
        model: Qwen/Qwen2.5-Coder-1.5B
        adapter_id: vladpp91/safeclaw-coder-lora
        device: cpu
        dtype: float32

    task_providers:
      coding: safeclaw-coder
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TransformersAdapterConfig:
    """Settings for the local transformers provider."""

    base_model: str
    adapter_id: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    trust_remote_code: bool = True
    hf_token: str | None = None
    extra_load_kwargs: dict[str, Any] = field(default_factory=dict)


class _TransformersAdapterRuntime:
    """Lazy holder for the loaded HF model + tokenizer."""

    def __init__(self, config: TransformersAdapterConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._model: Any = None
        self._tokenizer: Any = None

    @staticmethod
    def _resolve_dtype(dtype: str) -> Any:
        import torch  # lazy

        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        return mapping.get(dtype.lower(), torch.float32)

    def ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return self._model, self._tokenizer

            try:
                import torch  # noqa: F401  # ensures torch is installed
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - optional dep
                raise RuntimeError(
                    "Local transformers provider requires `transformers` and "
                    "`torch`. Install via `pip install safeclaw[finetune]`."
                ) from exc

            cfg = self.config
            token = cfg.hf_token or os.environ.get("HF_TOKEN") or None

            logger.info("Loading base model %s on %s (dtype=%s)",
                        cfg.base_model, cfg.device, cfg.dtype)
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.base_model,
                trust_remote_code=cfg.trust_remote_code,
                token=token,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            load_kwargs: dict[str, Any] = {
                "trust_remote_code": cfg.trust_remote_code,
                "torch_dtype": self._resolve_dtype(cfg.dtype),
                "low_cpu_mem_usage": True,
                "token": token,
            }
            load_kwargs.update(cfg.extra_load_kwargs)

            model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model,
                **load_kwargs,
            )

            if cfg.adapter_id:
                try:
                    from peft import PeftModel
                except ImportError as exc:  # pragma: no cover - optional dep
                    raise RuntimeError(
                        "Loading a LoRA adapter requires `peft`. Install via "
                        "`pip install safeclaw[finetune]`."
                    ) from exc

                logger.info("Attaching LoRA adapter %s", cfg.adapter_id)
                model = PeftModel.from_pretrained(
                    model,
                    cfg.adapter_id,
                    token=token,
                )

            model = model.to(cfg.device)
            model.eval()

            self._model = model
            self._tokenizer = tokenizer
            return self._model, self._tokenizer

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        import torch

        model, tokenizer = self.ensure_loaded()

        # Prefer the model's chat template when one is provided —
        # Qwen2.5-Coder ships an instruct-style template.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # pragma: no cover - fall back for base models
            text = f"{system_prompt}\n\n{prompt}\n"

        inputs = tokenizer(text, return_tensors="pt").to(self.config.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = 0.95

        with torch.inference_mode():
            output = model.generate(**inputs, **gen_kwargs)

        # Strip the prompt prefix from the generated tokens.
        new_tokens = output[0, inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


_RUNTIME_CACHE: dict[str, _TransformersAdapterRuntime] = {}


def _runtime_key(config: TransformersAdapterConfig) -> str:
    return f"{config.base_model}|{config.adapter_id}|{config.device}|{config.dtype}"


def get_runtime(config: TransformersAdapterConfig) -> _TransformersAdapterRuntime:
    """Return a process-wide cached runtime for the given config."""
    key = _runtime_key(config)
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = _TransformersAdapterRuntime(config)
        _RUNTIME_CACHE[key] = runtime
    return runtime


async def generate_async(
    config: TransformersAdapterConfig,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Run blocking HF generation off the event loop."""
    runtime = get_runtime(config)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        runtime.generate,
        prompt,
        system_prompt,
        temperature,
        max_tokens,
    )
