"""SFT training with QLoRA.

Fine-tunes a base model on JSON extraction using 4-bit quantization
and LoRA adapters. Designed to run on a single T4/A10 GPU.
"""

import logging
import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from prepare_data import format_chat_for_sft

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "../config/sft_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def create_quantization_config(cfg: dict) -> BitsAndBytesConfig:
    qcfg = cfg["quantization"]
    compute_dtype = getattr(torch, qcfg["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
    )


def create_lora_config(cfg: dict) -> LoraConfig:
    lcfg = cfg["lora"]
    return LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        target_modules=lcfg["target_modules"],
        task_type=lcfg["task_type"],
        bias=lcfg["bias"],
    )


def main():
    config_path = os.environ.get("SFT_CONFIG", "../config/sft_config.yaml")
    cfg = load_config(config_path)
    mcfg = cfg["model"]
    tcfg = cfg["training"]
    dcfg = cfg["data"]
    ocfg = cfg["output"]

    logger.info("Loading base model: %s", mcfg["base_model"])
    bnb_config = create_quantization_config(cfg)

    model = AutoModelForCausalLM.from_pretrained(
        mcfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(
        mcfg["base_model"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = create_lora_config(cfg)
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )

    logger.info("Loading dataset: %s", dcfg["sft_dataset"])
    train_ds = load_dataset(dcfg["sft_dataset"], split=dcfg["sft_split"])
    eval_ds = load_dataset(dcfg["sft_dataset"], split=dcfg["eval_split"])

    if dcfg.get("max_train_samples"):
        train_ds = train_ds.shuffle(seed=42).select(
            range(min(dcfg["max_train_samples"], len(train_ds)))
        )
    if dcfg.get("max_eval_samples"):
        eval_ds = eval_ds.shuffle(seed=42).select(
            range(min(dcfg["max_eval_samples"], len(eval_ds)))
        )

    train_ds = train_ds.map(format_chat_for_sft, remove_columns=train_ds.column_names)
    train_ds = train_ds.filter(lambda x: len(x.get("messages", [])) > 0)
    eval_ds = eval_ds.map(format_chat_for_sft, remove_columns=eval_ds.column_names)
    eval_ds = eval_ds.filter(lambda x: len(x.get("messages", [])) > 0)

    logger.info("Train: %d examples, Eval: %d examples", len(train_ds), len(eval_ds))

    training_args = TrainingArguments(
        output_dir=ocfg["output_dir"],
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        warmup_ratio=tcfg["warmup_ratio"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        logging_steps=tcfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=tcfg["eval_steps"],
        save_strategy="steps",
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        bf16=tcfg["bf16"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        optim=tcfg["optim"],
        report_to=tcfg["report_to"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        max_seq_length=tcfg["max_seq_length"],
    )

    logger.info("Starting SFT training...")
    train_result = trainer.train()

    logger.info("Saving adapter to %s", ocfg["adapter_dir"])
    trainer.save_model(ocfg["adapter_dir"])
    tokenizer.save_pretrained(ocfg["adapter_dir"])

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(eval_ds)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("SFT training complete. Adapter saved to: %s", ocfg["adapter_dir"])
    logger.info("Final train loss: %.4f", metrics.get("train_loss", 0))
    logger.info("Final eval loss: %.4f", eval_metrics.get("eval_loss", 0))


if __name__ == "__main__":
    main()
