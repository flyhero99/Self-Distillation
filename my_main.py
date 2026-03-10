import argparse
import json
import os
from string import Template
from typing import Any, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from distil_config import DistilConfig
from distil_trainer import DistilTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="OPSD training for instruction/thinking/response data.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct", help="Student model name/path.")
    parser.add_argument(
        "--teacher_model_name",
        type=str,
        default=None,
        help="Teacher model name/path. If unset, use --model_name.",
    )
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training JSON data.")
    parser.add_argument("--eval_data_path", type=str, default=None, help="Optional path to eval JSON data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument(
        "--num_prompts_per_batch",
        type=int,
        default=8,
        help="Effective prompts per update via gradient accumulation.",
    )
    parser.add_argument("--num_generations", type=int, default=8, help="Number of sampled completions per prompt.")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="Reference model mixup alpha.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_prompt_length", type=int, default=8192, help="Max prompt token length.")
    parser.add_argument("--max_completion_length", type=int, default=2048, help="Max completion token length.")
    parser.add_argument("--generate_from_teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report_to", type=str, default="wandb", help="Logging backend.")
    parser.add_argument("--num_loss_tokens_to_skip", type=int, default=3)
    parser.add_argument(
        "--debug_preview_num",
        type=int,
        default=3,
        help="Number of samples to print for prompt/output debugging before training.",
    )
    parser.add_argument(
        "--debug_preview_max_chars",
        type=int,
        default=1200,
        help="Max characters to print per preview field.",
    )
    return parser.parse_args()


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(_to_text(v) for v in value if v is not None).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2).strip()
    return str(value).strip()


def _load_json_or_jsonl(path: str) -> Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    suffix = os.path.splitext(path)[1].lower()
    if suffix not in {".json", ".jsonl"}:
        raise ValueError(f"Unsupported data suffix: {suffix}. Expected .json or .jsonl")
    # HF json loader supports both JSON arrays and JSONL; this keeps input handling consistent.
    return load_dataset("json", data_files=path, split="train")


def _build_teacher_content(instruction: str, thinking: str, response: str) -> str:
    teacher_prompt = Template("""
$instruction

This is an example for a response to the question:
$response

Now answer with a response of your own, including the thinking process.
""")
    return teacher_prompt.substitute(instruction=instruction, response=response)


def _truncate_for_preview(text: str, max_chars: int) -> str:
    return text


def _print_dataset_preview(dataset: Dataset, num_samples: int, max_chars: int, split_name: str) -> None:
    if num_samples <= 0 or len(dataset) == 0:
        return
    preview_count = min(num_samples, len(dataset))
    print(f"\n===== Debug Preview ({split_name}): showing {preview_count} sample(s) =====")
    for i in range(preview_count):
        row = dataset[i]
        instruction = _to_text(row.get("instruction"))
        thinking = _to_text(row.get("thinking"))
        # output_text = _to_text(row.get("program") if row.get("program") is not None else row.get("response"))
        output_text = _to_text(row.get("response"))
        student_prompt = instruction
        teacher_prompt = _build_teacher_content(instruction, thinking, output_text)
        print(f"\n----- Sample #{i} -----")
        print("[Student Prompt]")
        print(_truncate_for_preview(student_prompt, max_chars))
        print("\n[Teacher Prompt]")
        print(_truncate_for_preview(teacher_prompt, max_chars))
        print("\n[Example Output]")
        print(_truncate_for_preview(output_text, max_chars))
    print("===== End Debug Preview =====\n")


def load_codegen_dataset(
    train_path: str,
    eval_path: Optional[str],
    seed: int,
    debug_preview_num: int = 0,
    debug_preview_max_chars: int = 1200,
):
    train_dataset = _load_json_or_jsonl(train_path)
    eval_dataset = _load_json_or_jsonl(eval_path) if eval_path else None

    _print_dataset_preview(train_dataset, debug_preview_num, debug_preview_max_chars, split_name="train")
    if eval_dataset is not None:
        _print_dataset_preview(eval_dataset, min(1, debug_preview_num), debug_preview_max_chars, split_name="eval")

    def format_example(example):
        instruction = _to_text(example.get("instruction"))
        thinking = _to_text(example.get("thinking"))
        # Prefer pure code field if present in JSONL, otherwise fall back to response text.
        # response = _to_text(example.get("program") if example.get("program") is not None else example.get("response"))
        response = _to_text(example.get("response"))

        return {
            "prompt": [{"role": "user", "content": instruction}],
            "teacher_prompt": [
                {
                    "role": "user",
                    "content": _build_teacher_content(instruction, thinking, response),
                }
            ],
        }

    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names).shuffle(seed=seed)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(format_example, remove_columns=eval_dataset.column_names)
    return train_dataset, eval_dataset


if __name__ == "__main__":
    args = parse_args()
    teacher_model_name = args.teacher_model_name or args.model_name

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset, eval_dataset = load_codegen_dataset(
        args.train_data_path,
        args.eval_data_path,
        args.seed,
        debug_preview_num=args.debug_preview_num,
        debug_preview_max_chars=args.debug_preview_max_chars,
    )

    config = DistilConfig(
        seed=args.seed,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=True,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_prompts_per_batch,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        save_steps=100,
        max_grad_norm=1,
        report_to=args.report_to,
        output_dir=args.output_dir,
        log_completions=False,
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction=True,
        num_loss_tokens_to_skip=args.num_loss_tokens_to_skip,
        generate_from_teacher=args.generate_from_teacher,
    )

    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
