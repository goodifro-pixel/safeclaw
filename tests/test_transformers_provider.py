"""Tests for the local transformers / LoRA provider integration.

These do not load any real model — they only verify config parsing
and that ``${VAR}`` substitution from ``ai_writer.from_config`` works.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src"


def _load_ai_writer():
    """Load ``ai_writer`` directly to avoid pulling in the rest of safeclaw."""
    pkg_root = SRC / "safeclaw"
    # Register a minimal package so the relative import ``safeclaw.core...``
    # works inside ``_call_transformers``.
    if "safeclaw" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "safeclaw", pkg_root / "__init__.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["safeclaw"] = module
    if "safeclaw.core" not in sys.modules:
        core_init = pkg_root / "core" / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "safeclaw.core", core_init
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["safeclaw.core"] = module

    spec = importlib.util.spec_from_file_location(
        "safeclaw.core.ai_writer",
        pkg_root / "core" / "ai_writer.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["safeclaw.core.ai_writer"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ai_writer_module():
    return _load_ai_writer()


def test_transformers_provider_in_enum(ai_writer_module):
    providers = ai_writer_module.AIProvider
    assert providers.TRANSFORMERS.value == "transformers"
    assert providers.OPENROUTER.value == "openrouter"


def test_from_config_parses_transformers_block(ai_writer_module):
    cfg = {
        "ai_providers": [
            {
                "label": "safeclaw-coder",
                "provider": "transformers",
                "model": "Qwen/Qwen2.5-Coder-1.5B",
                "adapter_id": "vladpp91/safeclaw-coder-lora",
                "device": "cpu",
                "dtype": "float32",
            }
        ]
    }
    writer = ai_writer_module.AIWriter.from_config(cfg)
    provider = writer.providers["safeclaw-coder"]
    assert provider.provider is ai_writer_module.AIProvider.TRANSFORMERS
    assert provider.adapter_id == "vladpp91/safeclaw-coder-lora"
    assert provider.device == "cpu"
    assert provider.dtype == "float32"
    assert provider.model == "Qwen/Qwen2.5-Coder-1.5B"


def test_env_substitution_in_api_key(ai_writer_module, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-token")
    cfg = {
        "ai_providers": [
            {
                "label": "openrouter",
                "provider": "openrouter",
                "api_key": "${OPENROUTER_API_KEY}",
                "model": "meta-llama/llama-3.1-8b-instruct",
            }
        ]
    }
    writer = ai_writer_module.AIWriter.from_config(cfg)
    provider = writer.providers["openrouter"]
    assert provider.api_key == "sk-or-v1-test-token"
    assert provider.endpoint.startswith("https://openrouter.ai/")


def test_env_substitution_missing_var_yields_empty(ai_writer_module, monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET_VAR", raising=False)
    cfg = {
        "ai_providers": [
            {
                "label": "x",
                "provider": "openrouter",
                "api_key": "${DEFINITELY_NOT_SET_VAR}",
            }
        ]
    }
    writer = ai_writer_module.AIWriter.from_config(cfg)
    assert writer.providers["x"].api_key == ""


def test_list_providers_marks_transformers_as_local(ai_writer_module):
    cfg = {
        "ai_providers": [
            {
                "label": "safeclaw-coder",
                "provider": "transformers",
                "model": "Qwen/Qwen2.5-Coder-1.5B",
            }
        ]
    }
    writer = ai_writer_module.AIWriter.from_config(cfg)
    info = writer.list_providers()
    assert info[0]["is_local"] is True
    assert info[0]["adapter_id"] == ""
