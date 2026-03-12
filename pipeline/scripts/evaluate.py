"""Evaluation pipeline.

Compares base model vs SFT vs SFT+DPO on JSON extraction tasks.
Outputs metrics JSON for the showcase site.
"""

import json
import logging
import os
import re
import time

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prepare_data import JSON_EXTRACTION_SYSTEM, prepare_eval_prompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    base_model: str,
    adapter_path: str | None = None,
    use_4bit: bool = True,
):
    """Load model with optional adapter."""
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        logger.info("Loading adapter: %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def generate_response(
    model, tokenizer, prompt: str, max_new_tokens: int = 512
) -> tuple[str, float]:
    """Generate a response and measure latency."""
    messages = [
        {"role": "system", "content": JSON_EXTRACTION_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.perf_counter() - start

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response, elapsed


def extract_json(text: str) -> dict | None:
    """Try to parse JSON from model output."""
    text = text.strip()
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        text = json_match.group(1).strip()

    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        text = json_match.group(0)

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def score_extraction(predicted: dict | None, expected: dict) -> dict:
    """Score a single JSON extraction result."""
    if predicted is None:
        return {
            "valid_json": False,
            "key_precision": 0.0,
            "key_recall": 0.0,
            "key_f1": 0.0,
            "value_accuracy": 0.0,
        }

    expected_keys = set(expected.keys())
    predicted_keys = set(predicted.keys())

    true_pos = expected_keys & predicted_keys
    precision = len(true_pos) / len(predicted_keys) if predicted_keys else 0
    recall = len(true_pos) / len(expected_keys) if expected_keys else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    value_matches = 0
    for key in true_pos:
        exp_val = expected[key]
        pred_val = predicted.get(key)
        if isinstance(exp_val, list) and isinstance(pred_val, list):
            if set(map(str, exp_val)) == set(map(str, pred_val)):
                value_matches += 1
        elif str(exp_val).lower() == str(pred_val).lower():
            value_matches += 1

    value_accuracy = value_matches / len(expected_keys) if expected_keys else 0

    return {
        "valid_json": True,
        "key_precision": round(precision, 4),
        "key_recall": round(recall, 4),
        "key_f1": round(f1, 4),
        "value_accuracy": round(value_accuracy, 4),
    }


def evaluate_model(
    model, tokenizer, eval_prompts: list[dict], model_label: str
) -> dict:
    """Run full evaluation on a model."""
    logger.info("Evaluating: %s (%d prompts)", model_label, len(eval_prompts))

    results = []
    total_latency = 0
    total_tokens = 0

    for prompt_data in eval_prompts:
        response, latency = generate_response(model, tokenizer, prompt_data["input"])
        parsed = extract_json(response)
        score = score_extraction(parsed, prompt_data["expected"])

        token_count = len(tokenizer.encode(response))
        total_latency += latency
        total_tokens += token_count

        results.append(
            {
                "id": prompt_data["id"],
                "input": prompt_data["input"],
                "raw_output": response,
                "parsed_json": parsed,
                "expected": prompt_data["expected"],
                "scores": score,
                "latency_s": round(latency, 3),
                "tokens": token_count,
            }
        )

    valid_json_rate = sum(1 for r in results if r["scores"]["valid_json"]) / len(
        results
    )
    avg_key_f1 = sum(r["scores"]["key_f1"] for r in results) / len(results)
    avg_value_acc = sum(r["scores"]["value_accuracy"] for r in results) / len(results)
    avg_latency = total_latency / len(results)
    tokens_per_sec = total_tokens / total_latency if total_latency > 0 else 0

    summary = {
        "model": model_label,
        "num_prompts": len(results),
        "valid_json_rate": round(valid_json_rate, 4),
        "avg_key_f1": round(avg_key_f1, 4),
        "avg_value_accuracy": round(avg_value_acc, 4),
        "avg_latency_s": round(avg_latency, 3),
        "tokens_per_second": round(tokens_per_sec, 1),
        "total_tokens": total_tokens,
    }

    logger.info(
        "%s — JSON: %.1f%% | Key F1: %.3f | Value Acc: %.3f | Latency: %.2fs",
        model_label,
        valid_json_rate * 100,
        avg_key_f1,
        avg_value_acc,
        avg_latency,
    )

    return {"summary": summary, "results": results}


def run_full_evaluation(
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    sft_adapter: str = "./outputs/sft/final_adapter",
    dpo_adapter: str = "./outputs/dpo/final_adapter",
    output_path: str = "../results/eval_metrics.json",
):
    """Run evaluation across base, SFT, and SFT+DPO models."""
    eval_prompts = prepare_eval_prompts()
    all_results = {}

    logger.info("=== Evaluating Base Model ===")
    model, tokenizer = load_model_and_tokenizer(base_model)
    all_results["base"] = evaluate_model(model, tokenizer, eval_prompts, "Base Model")
    del model
    torch.cuda.empty_cache()

    if os.path.exists(sft_adapter):
        logger.info("=== Evaluating SFT Model ===")
        model, tokenizer = load_model_and_tokenizer(base_model, sft_adapter)
        all_results["sft"] = evaluate_model(
            model, tokenizer, eval_prompts, "SFT (QLoRA)"
        )
        del model
        torch.cuda.empty_cache()

    if os.path.exists(dpo_adapter):
        logger.info("=== Evaluating SFT + DPO Model ===")
        model, tokenizer = load_model_and_tokenizer(base_model, dpo_adapter)
        all_results["dpo"] = evaluate_model(
            model, tokenizer, eval_prompts, "SFT + DPO"
        )
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Evaluation results saved to: %s", output_path)

    logger.info("\n=== COMPARISON ===")
    header = f"{'Model':<20} {'JSON%':>8} {'Key F1':>8} {'Val Acc':>8} {'Latency':>8} {'Tok/s':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for key, data in all_results.items():
        s = data["summary"]
        logger.info(
            f"{s['model']:<20} {s['valid_json_rate']*100:>7.1f}% {s['avg_key_f1']:>8.3f} "
            f"{s['avg_value_accuracy']:>8.3f} {s['avg_latency_s']:>7.2f}s {s['tokens_per_second']:>7.1f}"
        )

    return all_results


if __name__ == "__main__":
    run_full_evaluation()
