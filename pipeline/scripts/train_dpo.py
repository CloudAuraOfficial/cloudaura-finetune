"""DPO preference tuning.

Applies Direct Preference Optimization on top of the SFT adapter
to align model outputs with human preferences.
"""

import logging
import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer

from prepare_data import prepare_dpo_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "../config/dpo_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    config_path = os.environ.get("DPO_CONFIG", "../config/dpo_config.yaml")
    cfg = load_config(config_path)
    mcfg = cfg["model"]
    tcfg = cfg["training"]
    dcfg = cfg["data"]
    ocfg = cfg["output"]
    qcfg = cfg["quantization"]
    lcfg = cfg["lora"]

    compute_dtype = getattr(torch, qcfg["bnb_4bit_compute_dtype"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
    )

    logger.info("Loading base model: %s", mcfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Loading SFT adapter: %s", mcfg["sft_adapter"])
    model = PeftModel.from_pretrained(model, mcfg["sft_adapter"], is_trainable=False)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(
        mcfg["base_model"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        target_modules=lcfg["target_modules"],
        task_type=lcfg["task_type"],
        bias=lcfg["bias"],
    )

    logger.info("Preparing DPO dataset...")
    dpo_ds = prepare_dpo_data(
        dataset_name=dcfg["dataset"],
        max_samples=dcfg.get("max_train_samples", 5000),
    )

    split = dpo_ds.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    logger.info("DPO Train: %d, Eval: %d", len(train_ds), len(eval_ds))

    dpo_args = DPOConfig(
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
        beta=tcfg["beta"],
        loss_type=tcfg["loss_type"],
        max_length=tcfg["max_length"],
        max_prompt_length=tcfg["max_prompt_length"],
        report_to=tcfg["report_to"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    logger.info("Starting DPO training...")
    train_result = trainer.train()

    logger.info("Saving DPO adapter to %s", ocfg["adapter_dir"])
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

    logger.info("DPO training complete. Adapter saved to: %s", ocfg["adapter_dir"])
    logger.info("Final train loss: %.4f", metrics.get("train_loss", 0))
    logger.info("Final eval loss: %.4f", eval_metrics.get("eval_loss", 0))


if __name__ == "__main__":
    main()
