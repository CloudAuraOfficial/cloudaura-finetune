"""Data preparation for SFT and DPO training.

Loads datasets from Hugging Face, formats them for the target task
(JSON extraction), and creates train/eval splits.
"""

import json
import logging

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JSON_EXTRACTION_SYSTEM = (
    "You are a precise JSON extraction assistant. Given a user's natural language "
    "input, extract the relevant information and return it as a valid JSON object. "
    "Only output JSON — no explanations, no markdown fences."
)

JSON_EXTRACTION_EXAMPLES = [
    {
        "input": "Book a flight from New York to London on March 15th for 2 adults in economy class.",
        "output": json.dumps(
            {
                "action": "book_flight",
                "origin": "New York",
                "destination": "London",
                "date": "2025-03-15",
                "passengers": 2,
                "class": "economy",
            }
        ),
    },
    {
        "input": "Schedule a meeting with Sarah and Tom tomorrow at 3pm in Conference Room B about Q4 planning.",
        "output": json.dumps(
            {
                "action": "schedule_meeting",
                "attendees": ["Sarah", "Tom"],
                "date": "tomorrow",
                "time": "15:00",
                "location": "Conference Room B",
                "subject": "Q4 planning",
            }
        ),
    },
    {
        "input": "Create a new user account for jane.doe@company.com with admin role in the billing department.",
        "output": json.dumps(
            {
                "action": "create_user",
                "email": "jane.doe@company.com",
                "role": "admin",
                "department": "billing",
            }
        ),
    },
    {
        "input": "Send an invoice for $2,500 to Acme Corp for consulting services rendered in January.",
        "output": json.dumps(
            {
                "action": "send_invoice",
                "amount": 2500,
                "currency": "USD",
                "recipient": "Acme Corp",
                "description": "consulting services",
                "period": "January",
            }
        ),
    },
    {
        "input": "Update the shipping address for order #12345 to 742 Evergreen Terrace, Springfield, IL 62704.",
        "output": json.dumps(
            {
                "action": "update_address",
                "order_id": "12345",
                "street": "742 Evergreen Terrace",
                "city": "Springfield",
                "state": "IL",
                "zip": "62704",
            }
        ),
    },
]


def format_chat_for_sft(example: dict) -> dict:
    """Convert ultrachat conversation to JSON extraction training format."""
    messages = example.get("messages", [])
    if len(messages) < 2:
        return {"text": ""}

    user_msg = messages[0].get("content", "") if messages[0]["role"] == "user" else ""
    assistant_msg = (
        messages[1].get("content", "") if messages[1]["role"] == "assistant" else ""
    )

    if not user_msg or not assistant_msg:
        return {"text": ""}

    formatted = [
        {"role": "system", "content": JSON_EXTRACTION_SYSTEM},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    return {"messages": formatted}


def format_json_extraction_examples() -> list[dict]:
    """Create formatted JSON extraction training examples."""
    formatted = []
    for ex in JSON_EXTRACTION_EXAMPLES:
        formatted.append(
            {
                "messages": [
                    {"role": "system", "content": JSON_EXTRACTION_SYSTEM},
                    {"role": "user", "content": ex["input"]},
                    {"role": "assistant", "content": ex["output"]},
                ]
            }
        )
    return formatted


def prepare_sft_data(
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    split: str = "train_sft",
    max_samples: int = 10000,
) -> dict:
    """Load and prepare SFT dataset."""
    logger.info("Loading SFT dataset: %s (split=%s)", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)

    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(max_samples))
        logger.info("Sampled %d examples", max_samples)

    ds = ds.map(format_chat_for_sft, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x.get("messages", [])) > 0)

    logger.info("SFT dataset prepared: %d examples", len(ds))
    return ds


def prepare_dpo_data(
    dataset_name: str = "argilla/distilabel-intel-orca-dpo-pairs",
    max_samples: int = 5000,
) -> dict:
    """Load and prepare DPO preference dataset."""
    logger.info("Loading DPO dataset: %s", dataset_name)
    ds = load_dataset(dataset_name, split="train")

    ds = ds.filter(lambda x: x.get("status", "") != "tie")

    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(max_samples))
        logger.info("Sampled %d preference pairs", max_samples)

    def format_dpo(example):
        prompt = example.get("input", "")
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")
        return {
            "prompt": [
                {"role": "system", "content": JSON_EXTRACTION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        }

    ds = ds.map(format_dpo, remove_columns=ds.column_names)
    logger.info("DPO dataset prepared: %d preference pairs", len(ds))
    return ds


def prepare_eval_prompts() -> list[dict]:
    """Create evaluation prompts for JSON extraction task."""
    return [
        {
            "id": "eval_001",
            "input": "Cancel subscription for user ID 8842 effective immediately and send a confirmation email.",
            "expected": {
                "action": "cancel_subscription",
                "user_id": "8842",
                "effective": "immediately",
                "send_confirmation": True,
            },
        },
        {
            "id": "eval_002",
            "input": "Transfer $500 from checking account to savings account for customer John Smith.",
            "expected": {
                "action": "transfer_funds",
                "amount": 500,
                "from_account": "checking",
                "to_account": "savings",
                "customer": "John Smith",
            },
        },
        {
            "id": "eval_003",
            "input": "Deploy version 2.4.1 of the payment-service to production in us-east-1 with a 10% canary rollout.",
            "expected": {
                "action": "deploy",
                "service": "payment-service",
                "version": "2.4.1",
                "environment": "production",
                "region": "us-east-1",
                "rollout_strategy": "canary",
                "rollout_percentage": 10,
            },
        },
        {
            "id": "eval_004",
            "input": "Add a DNS A record for api.example.com pointing to 192.168.1.100 with TTL of 300 seconds.",
            "expected": {
                "action": "add_dns_record",
                "type": "A",
                "hostname": "api.example.com",
                "value": "192.168.1.100",
                "ttl": 300,
            },
        },
        {
            "id": "eval_005",
            "input": "Create a support ticket: priority high, category billing, assigned to finance team. Customer reports double charge on order #7891.",
            "expected": {
                "action": "create_ticket",
                "priority": "high",
                "category": "billing",
                "assigned_to": "finance team",
                "description": "double charge on order #7891",
            },
        },
        {
            "id": "eval_006",
            "input": "Register a webhook at https://hooks.example.com/payments for payment.completed and payment.failed events with HMAC-SHA256 signing.",
            "expected": {
                "action": "register_webhook",
                "url": "https://hooks.example.com/payments",
                "events": ["payment.completed", "payment.failed"],
                "signing": "HMAC-SHA256",
            },
        },
        {
            "id": "eval_007",
            "input": "Scale the worker pool to 8 instances with 4 vCPUs and 16GB RAM each in the analytics namespace.",
            "expected": {
                "action": "scale",
                "resource": "worker pool",
                "instances": 8,
                "vcpus": 4,
                "memory_gb": 16,
                "namespace": "analytics",
            },
        },
        {
            "id": "eval_008",
            "input": "Revoke API key ak_live_abc123 for application 'mobile-app' and notify the developer at dev@company.io.",
            "expected": {
                "action": "revoke_api_key",
                "key_id": "ak_live_abc123",
                "application": "mobile-app",
                "notify": "dev@company.io",
            },
        },
    ]


if __name__ == "__main__":
    sft_ds = prepare_sft_data(max_samples=100)
    logger.info("SFT sample: %s", sft_ds[0])

    dpo_ds = prepare_dpo_data(max_samples=100)
    logger.info("DPO sample: %s", dpo_ds[0])

    eval_prompts = prepare_eval_prompts()
    logger.info("Eval prompts: %d", len(eval_prompts))
