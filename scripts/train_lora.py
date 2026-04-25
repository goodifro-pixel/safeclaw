#!/usr/bin/env python3
"""
LoRA fine-tuning proof-of-concept for SafeClaw's coding assistant.

Trains a LoRA adapter on top of Qwen/Qwen2.5-Coder-1.5B using
bigcode/the-stack-smol code samples (python, javascript, typescript,
go, rust). Designed to run on CPU only — slow but reproducible — and
push the resulting adapter to the HuggingFace Hub.

Usage:
    HF_TOKEN=hf_... python scripts/train_lora.py \\
        --output-dir checkpoints/safeclaw-coder-lora \\
        --hub-model-id goodifro-pixel/safeclaw-coder-lora \\
        --num-samples 5000 \\
        --max-length 1024 \\
        --epochs 1

Notes:
    * `bigcode/the-stack-smol` is gated. Accept the dataset license at
      https://huggingface.co/datasets/bigcode/the-stack-smol once with
      the same account whose token is used here.
    * On CPU the training run for the default settings is expected to
      take many hours. Reduce ``--num-samples`` for quicker smoke tests.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger("safeclaw.train_lora")

DEFAULT_LANGUAGES = ("python", "javascript", "typescript", "go", "rust")
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
DEFAULT_DATASET = "bigcode/the-stack-smol"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(DEFAULT_LANGUAGES),
        help="Programming language subsets to load from the dataset.",
    )
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        default="checkpoints/safeclaw-coder-lora",
        help="Local directory for checkpoints and the final adapter.",
    )
    parser.add_argument(
        "--hub-model-id",
        default="goodifro-pixel/safeclaw-coder-lora",
        help="HuggingFace Hub repo to push the adapter to (set empty to skip).",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the resulting adapter to the HF Hub at the end of training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Seed used to shuffle the combined multilingual dataset.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level for this script.",
    )
    return parser.parse_args()


def build_dataset(
    dataset_name: str,
    languages: list[str],
    num_samples: int,
    seed: int,
    token: str | None,
) -> Dataset:
    """Load a roughly-balanced multilingual coding dataset."""
    per_lang = max(1, num_samples // max(1, len(languages)))
    parts: list[Dataset] = []

    for lang in languages:
        logger.info("Loading %s split for language=%s", dataset_name, lang)
        ds = load_dataset(
            dataset_name,
            data_dir=f"data/{lang}",
            split="train",
            token=token,
        )
        if len(ds) > per_lang:
            ds = ds.shuffle(seed=seed).select(range(per_lang))
        parts.append(ds)

    combined = concatenate_datasets(parts).shuffle(seed=seed)
    if len(combined) > num_samples:
        combined = combined.select(range(num_samples))
    logger.info("Combined dataset: %d examples across %s", len(combined), languages)
    return combined


def format_example(example: dict, languages: set[str]) -> dict:
    """Map a raw the-stack-smol row into a SFT-friendly text field."""
    lang = (example.get("lang") or example.get("language") or "").lower()
    if lang not in languages:
        # Fall back to file-extension-based hint when missing.
        path = example.get("max_stars_repo_path") or example.get("path") or ""
        ext = Path(path).suffix.lower()
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".go": "go",
            ".rs": "rust",
        }
        lang = ext_map.get(ext, lang or "code")
    content = example.get("content") or example.get("text") or ""
    return {"text": f"<|fim_prefix|>// language: {lang}\n{content}<|fim_suffix|>"}


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    hf_token = os.environ.get("HF_TOKEN") or None
    if args.push_to_hub and not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is required when --push-to-hub is set."
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_ds = build_dataset(
        dataset_name=args.dataset,
        languages=args.languages,
        num_samples=args.num_samples,
        seed=args.seed,
        token=hf_token,
    )
    languages_lower = {lang.lower() for lang in args.languages}
    train_ds = raw_ds.map(
        lambda ex: format_example(ex, languages_lower),
        remove_columns=[c for c in raw_ds.column_names if c != "text"],
        desc="Formatting examples for SFT",
    )

    logger.info("Loading tokenizer + base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CPU-only: keep weights in float32 (bf16 on x86 is fine but optimizer
    # states still upcast). Disable FA/flash kernels.
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        token=hf_token,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        bf16=False,
        fp16=False,
        optim="adamw_torch",
        max_seq_length=args.max_length,
        dataset_text_field="text",
        packing=False,
        gradient_checkpointing=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id or None,
        hub_token=hf_token,
        hub_private_repo=False,
        seed=args.seed,
    )

    # Some TRL versions of SFTConfig keep TrainingArguments fields,
    # double-check they survive serialization.
    assert isinstance(sft_config, TrainingArguments)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )

    logger.info("Starting training: %d examples, max_length=%d, epochs=%s",
                len(train_ds), args.max_length, args.epochs)
    trainer.train()

    logger.info("Saving final adapter to %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if args.push_to_hub and args.hub_model_id:
        logger.info("Pushing adapter to HF Hub: %s", args.hub_model_id)
        trainer.push_to_hub()
        tokenizer.push_to_hub(args.hub_model_id, token=hf_token)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
