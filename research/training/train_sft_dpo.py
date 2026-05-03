"""
DPO fine-tuning on top of the SFT split-generator adapter.

Why this works now (but failed before):
  The original DPO attempt used Mistral-7B-Instruct as the reference model.
  The target format (CIDI JSON) was far from Mistral's base distribution, so
  both chosen and rejected logps diverged together (classic DPO collapse).

  The SFT adapter now puts the model *inside* the valid-JSON distribution
  (96.6% structural validity). DPO can now push chosen splits toward higher
  CDI and rejected toward lower, without escaping the format.

Strategy:
  1. Load Mistral-7B base
  2. Load SFT LoRA → merge into base (creates a 7B "SFT model" in one weight tensor)
  3. Use merged model as BOTH starting point and frozen reference model
  4. Apply DPO with a new LoRA on top (standard SFT→DPO pipeline)

Data: same split_dpo_{train,test}.jsonl — chosen = CDI>=0.5, rejected = CDI<0.5.

Hyperparams:
  beta=0.1   — moderate KL constraint (ref is SFT, not raw Mistral, so can afford higher)
  lr=5e-6    — smaller than SFT; DPO is sensitive to overfitting
  epochs=1   — DPO needs fewer epochs; the format is already learned
"""
from __future__ import annotations
import argparse, json, copy
from pathlib import Path

TRAIN_JSONL  = Path("outputs/training/split_dpo_train.jsonl")
TEST_JSONL   = Path("outputs/training/split_dpo_test.jsonl")
SFT_ADAPTER  = Path("outputs/training/split_adapter/final_adapter")
OUTPUT_DIR   = Path("outputs/training/sft_dpo_adapter")
DEFAULT_BASE = "/scratch/mir85108/llm/models/hf/Mistral-7B-Instruct-v0.3"


def _load_dpo_dataset(path: Path, tokenizer):
    from datasets import Dataset
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chosen_text   = tokenizer.apply_chat_template(
                d["chosen"], tokenize=False, add_generation_prompt=False
            )
            rejected_text = tokenizer.apply_chat_template(
                d["rejected"], tokenize=False, add_generation_prompt=False
            )
            prompt_msgs   = [m for m in d["chosen"] if m["role"] != "assistant"]
            prompt_text   = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            records.append({
                "prompt":   prompt_text,
                "chosen":   chosen_text,
                "rejected": rejected_text,
            })
    return Dataset.from_list(records)


def train(
    base_model:   str   = DEFAULT_BASE,
    sft_adapter:  Path  = SFT_ADAPTER,
    output_dir:   Path  = OUTPUT_DIR,
    epochs:       int   = 1,
    lr:           float = 5e-6,
    beta:         float = 0.1,
    batch_size:   int   = 1,
    grad_accum:   int   = 8,
    max_length:   int   = 2048,
    lora_r:       int   = 16,
) -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, PeftModel
    from trl import DPOTrainer, DPOConfig

    if not TRAIN_JSONL.exists():
        raise FileNotFoundError(f"{TRAIN_JSONL} not found.")
    if not Path(sft_adapter).exists():
        raise FileNotFoundError(f"SFT adapter not found at {sft_adapter}. Run SFT training first.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Base model   : {base_model}")
    print(f"[INFO] SFT adapter  : {sft_adapter}")
    print(f"[INFO] Method       : DPO on top of SFT (merged reference)")
    print(f"[INFO] beta         : {beta}")
    print(f"[INFO] Epochs       : {epochs}")
    print(f"[INFO] LR           : {lr}")
    print(f"[INFO] LoRA r/alpha : {lora_r}/{lora_r * 2}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Step 1: build merged SFT model (reference + starting point) ──────────
    print("[INFO] Loading base model and merging SFT adapter...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, trust_remote_code=True
    )
    sft_peft = PeftModel.from_pretrained(base, str(sft_adapter))
    merged   = sft_peft.merge_and_unload()   # 7B float16 model with SFT weights baked in
    merged.config.use_cache = False
    print("[INFO] SFT adapter merged into base weights.")

    # ── Step 2: reference model = frozen copy of merged ──────────────────────
    print("[INFO] Creating frozen reference model (copy of merged SFT)...")
    ref_model = copy.deepcopy(merged)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # Move both to CUDA
    merged    = merged.cuda()
    ref_model = ref_model.cuda()

    # ── Step 3: apply new LoRA on top of merged model ────────────────────────
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    train_ds = _load_dpo_dataset(TRAIN_JSONL, tokenizer)
    test_ds  = _load_dpo_dataset(TEST_JSONL, tokenizer) if TEST_JSONL.exists() else None
    print(f"[INFO] Train pairs  : {len(train_ds)}")
    if test_ds:
        print(f"[INFO] Test  pairs  : {len(test_ds)}")

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        beta=beta,
        fp16=True,
        logging_steps=5,
        eval_strategy="epoch" if test_ds else "no",
        save_strategy="epoch",
        save_total_limit=1,
        max_length=max_length,
        max_prompt_length=512,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=merged,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    print("[INFO] Starting SFT→DPO training...")
    trainer.train()

    adapter_path = Path(output_dir) / "final_adapter"
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"[INFO] DPO adapter saved → {adapter_path}")

    meta = {
        "base_model":  base_model,
        "sft_adapter": str(sft_adapter),
        "method":      "SFT→DPO",
        "beta":        beta,
        "epochs":      epochs,
        "lr":          lr,
        "lora_r":      lora_r,
        "train_pairs": len(train_ds),
        "test_pairs":  len(test_ds) if test_ds else 0,
    }
    (Path(output_dir) / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print("[INFO] SFT→DPO training complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",  default=DEFAULT_BASE)
    parser.add_argument("--sft-adapter", default=str(SFT_ADAPTER))
    parser.add_argument("--output-dir",  default=str(OUTPUT_DIR))
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--lr",          type=float, default=5e-6)
    parser.add_argument("--beta",        type=float, default=0.1)
    parser.add_argument("--batch-size",  type=int,   default=1)
    parser.add_argument("--grad-accum",  type=int,   default=8)
    parser.add_argument("--max-length",  type=int,   default=2048)
    parser.add_argument("--lora-r",      type=int,   default=16)
    args = parser.parse_args()
    train(
        base_model=args.base_model,
        sft_adapter=Path(args.sft_adapter),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        lora_r=args.lora_r,
    )


if __name__ == "__main__":
    main()
