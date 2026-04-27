"""
QLoRA fine-tuning of a compact splitter model on the CollabMath jigsaw dataset.

Base model: Qwen/Qwen2.5-7B-Instruct  (~14 GB fp16, ~4 GB in 4-bit)
            A 7B model is sufficient for the splitter task: it only needs to
            classify the problem structure and produce well-formed JSON.

The fine-tuned adapter (~50 MB) is saved alongside the base model weights
so the GPU inference job can merge and load it.

Prerequisites (already installed by sapelo_gpu_setup.sh):
  pip install transformers datasets peft trl bitsandbytes accelerate

Usage (run from the scratch working directory):
  python -m research.training.finetune_splitter
  python -m research.training.finetune_splitter --epochs 3 --base-model Qwen/Qwen2.5-14B-Instruct
"""
import argparse
import json
from pathlib import Path

TRAIN_JSONL  = Path("outputs/training/splits_instruct.jsonl")
OUTPUT_DIR   = Path("outputs/training/splitter_adapter")
DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"


def _load_dataset(jsonl_path: Path):
    from datasets import Dataset
    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def _format_chat(example: dict, tokenizer) -> dict:
    """Apply the model's chat template to produce a single 'text' field."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def train(
    base_model: str = DEFAULT_BASE,
    epochs: int = 2,
    lr: float = 2e-4,
    batch_size: int = 2,
    grad_accum: int = 8,
    max_seq_len: int = 3072,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    if not TRAIN_JSONL.exists():
        raise FileNotFoundError(
            f"{TRAIN_JSONL} not found. Run prepare_finetune_data.py first."
        )

    print(f"[INFO] Base model  : {base_model}")
    print(f"[INFO] Epochs      : {epochs}")
    print(f"[INFO] Batch×Accum : {batch_size}×{grad_accum} = {batch_size * grad_accum} eff.")
    print(f"[INFO] LoRA r/α    : {lora_r}/{lora_alpha}")
    print(f"[INFO] Output dir  : {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 4-bit quantization ────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA adapters ─────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    raw_ds = _load_dataset(TRAIN_JSONL)
    ds = raw_ds.map(
        lambda ex: _format_chat(ex, tokenizer),
        remove_columns=raw_ds.column_names,
    )
    print(f"[INFO] Training examples: {len(ds)}")

    # ── Train ─────────────────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        max_seq_length=max_seq_len,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=sft_config,
    )

    print("[INFO] Starting training ...")
    trainer.train()

    # ── Save adapter only ─────────────────────────────────────────────────────
    adapter_path = OUTPUT_DIR / "final_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"[INFO] Adapter saved to {adapter_path}")

    # Save training metadata
    meta = {
        "base_model":  base_model,
        "epochs":      epochs,
        "lora_r":      lora_r,
        "lora_alpha":  lora_alpha,
        "train_size":  len(ds),
    }
    with open(OUTPUT_DIR / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[INFO] Fine-tuning complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",  default=DEFAULT_BASE)
    parser.add_argument("--epochs",      type=int,   default=2)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--batch-size",  type=int,   default=2)
    parser.add_argument("--grad-accum",  type=int,   default=8)
    parser.add_argument("--max-seq-len", type=int,   default=3072)
    parser.add_argument("--lora-r",      type=int,   default=16)
    args = parser.parse_args()
    train(
        base_model=args.base_model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
    )


if __name__ == "__main__":
    main()
