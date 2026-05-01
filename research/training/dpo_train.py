"""
DPO fine-tuning of an open-source LLM to generate L3 (student-sim) collaborative behavior.

Base model: meta-llama/Llama-3.1-8B-Instruct  (or Mistral-7B-Instruct-v0.3)
Method:     Direct Preference Optimization (Rafailov et al. 2023) via TRL
Adapter:    QLoRA 4-bit (fits in ~14 GB VRAM; single A100-40 GB on Sapelo)

The training signal: conversations with CDI≥0.5 (deep collaboration) are preferred
over CDI<0.5 conversations on the same problem. The model learns to produce the
exploratory, question-rich first-turn style that drives higher CDI.

Prerequisites
-------------
  pip install transformers datasets peft trl bitsandbytes accelerate

Usage
-----
  python -m research.training.dpo_train
  python -m research.training.dpo_train --base-model mistralai/Mistral-7B-Instruct-v0.3
  python -m research.training.dpo_train --epochs 3 --beta 0.1
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

TRAIN_JSONL  = Path("outputs/training/dpo_train.jsonl")
TEST_JSONL   = Path("outputs/training/dpo_test.jsonl")
OUTPUT_DIR   = Path("outputs/training/dpo_adapter")
DEFAULT_BASE = "meta-llama/Llama-3.1-8B-Instruct"


def _load_jsonl(path: Path):
    from datasets import Dataset
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def train(
    base_model:  str   = DEFAULT_BASE,
    epochs:      int   = 2,
    lr:          float = 5e-5,
    beta:        float = 0.1,
    batch_size:  int   = 1,
    grad_accum:  int   = 8,
    max_length:  int   = 2048,
    lora_r:      int   = 16,
    lora_alpha:  int   = 32,
    lora_dropout: float = 0.05,
) -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import DPOTrainer, DPOConfig

    if not TRAIN_JSONL.exists():
        raise FileNotFoundError(
            f"{TRAIN_JSONL} not found. Run prepare_dpo_data.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Base model    : {base_model}")
    print(f"[INFO] DPO beta      : {beta}")
    print(f"[INFO] Epochs        : {epochs}")
    print(f"[INFO] Batch×Accum   : {batch_size}×{grad_accum} = {batch_size * grad_accum} eff.")
    print(f"[INFO] LoRA r / alpha: {lora_r} / {lora_alpha}")

    # ── 4-bit quantization ────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # DPO prefers left-padding

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        # No device_map: bitsandbytes 4-bit places layers on CUDA automatically.
        # device_map (any value) triggers accelerate's dispatch_model which calls
        # model.to(device) — unsupported on quantized models.
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # Reference model: same weights but frozen (no LoRA)
    model_ref = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

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

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = _load_jsonl(TRAIN_JSONL)
    test_ds  = _load_jsonl(TEST_JSONL) if TEST_JSONL.exists() else None
    print(f"[INFO] Train examples: {len(train_ds)}")
    if test_ds:
        print(f"[INFO] Test  examples: {len(test_ds)}")

    # ── DPO training config ───────────────────────────────────────────────────
    dpo_config = DPOConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        beta=beta,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=5,
        eval_strategy="epoch" if test_ds else "no",
        save_strategy="epoch",
        save_total_limit=2,
        max_length=max_length,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
    )

    print("[INFO] Starting DPO training ...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    adapter_path = OUTPUT_DIR / "final_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"[INFO] Adapter saved → {adapter_path}")

    meta = {
        "base_model":  base_model,
        "epochs":      epochs,
        "beta":        beta,
        "lora_r":      lora_r,
        "lora_alpha":  lora_alpha,
        "train_size":  len(train_ds),
        "test_size":   len(test_ds) if test_ds else 0,
    }
    (OUTPUT_DIR / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print("[INFO] DPO fine-tuning complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",   default=DEFAULT_BASE)
    parser.add_argument("--epochs",       type=int,   default=2)
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--beta",         type=float, default=0.1,
                        help="DPO temperature (higher → stronger preference learning)")
    parser.add_argument("--batch-size",   type=int,   default=1)
    parser.add_argument("--grad-accum",   type=int,   default=8)
    parser.add_argument("--max-length",   type=int,   default=2048)
    parser.add_argument("--lora-r",       type=int,   default=16)
    args = parser.parse_args()
    train(
        base_model=args.base_model,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
    )


if __name__ == "__main__":
    main()
