"""
SFT fine-tuning of Mistral-7B-Instruct to generate CIDI-quality jigsaw splits.

Switched from DPO to SFT because DPO collapsed (both chosen/rejected logps diverge
from reference while margins inflate — model escapes distribution rather than learning
the split format). SFT is appropriate here: the task is format learning, not preference
ranking between two plausible outputs.

Training signal: the 333 chosen (CIDI, CDI >= 0.5) splits from the existing DPO JSONL.
At inference: generate a split, evaluate CDI, filter (sample-then-filter pipeline).

Data: reuses outputs/training/split_dpo_{train,test}.jsonl — extracts "chosen" field only.
No need to regenerate or re-transfer data to Sapelo.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

TRAIN_JSONL  = Path("outputs/training/split_dpo_train.jsonl")
TEST_JSONL   = Path("outputs/training/split_dpo_test.jsonl")
OUTPUT_DIR   = Path("outputs/training/split_adapter")
DEFAULT_BASE = "meta-llama/Llama-3.1-8B-Instruct"


def _load_sft_dataset(path: Path, tokenizer, max_examples: int | None = None):
    from datasets import Dataset
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chosen_msgs = d["chosen"]  # [system, user, assistant]
            text = tokenizer.apply_chat_template(
                chosen_msgs, tokenize=False, add_generation_prompt=False
            )
            records.append({"text": text})
    if max_examples is not None:
        records = records[:max_examples]
    return Dataset.from_list(records)


def train(
    base_model:    str   = DEFAULT_BASE,
    epochs:        int   = 3,
    lr:            float = 2e-5,
    batch_size:    int   = 1,
    grad_accum:    int   = 8,
    max_length:    int   = 2048,
    lora_r:        int   = 16,
    lora_alpha:    int   = 32,
    lora_dropout:  float = 0.05,
    max_examples:  int | None = None,
    output_dir:    Path  = OUTPUT_DIR,
) -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    if not TRAIN_JSONL.exists():
        raise FileNotFoundError(f"{TRAIN_JSONL} not found.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Base model    : {base_model}")
    print(f"[INFO] Method        : SFT (chosen splits only, no reference model)")
    print(f"[INFO] Epochs        : {epochs}")
    print(f"[INFO] LR            : {lr}")
    print(f"[INFO] Batch×Accum   : {batch_size}×{grad_accum} = {batch_size * grad_accum} eff.")
    print(f"[INFO] LoRA r / alpha: {lora_r} / {lora_alpha}")
    print(f"[INFO] Max length    : {max_length}")
    print(f"[INFO] Max examples  : {max_examples if max_examples else 'all'}")
    print(f"[INFO] Precision     : float16 (no quantization — fits A100-40GB)")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).cuda()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    train_ds = _load_sft_dataset(TRAIN_JSONL, tokenizer, max_examples=max_examples)
    test_ds  = _load_sft_dataset(TEST_JSONL, tokenizer) if TEST_JSONL.exists() else None
    print(f"[INFO] Train examples: {len(train_ds)}")
    if test_ds:
        print(f"[INFO] Test  examples: {len(test_ds)}")

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=5,
        eval_strategy="epoch" if test_ds else "no",
        save_strategy="epoch",
        save_total_limit=2,
        max_seq_length=max_length,
        dataset_text_field="text",
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    print("[INFO] Starting SFT training (split generator)...")
    trainer.train()

    adapter_path = output_dir / "final_adapter"
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"[INFO] Adapter saved → {adapter_path}")

    meta = {
        "base_model":   base_model,
        "method":       "SFT",
        "task":         "split_generator",
        "precision":    "float16",
        "epochs":       epochs,
        "lr":           lr,
        "lora_r":       lora_r,
        "lora_alpha":   lora_alpha,
        "train_size":   len(train_ds),
        "test_size":    len(test_ds) if test_ds else 0,
        "max_examples": max_examples,
    }
    (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print("[INFO] Split generator SFT fine-tuning complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",  default=DEFAULT_BASE)
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--batch-size",  type=int,   default=1)
    parser.add_argument("--grad-accum",  type=int,   default=8)
    parser.add_argument("--max-length",  type=int,   default=2048)
    parser.add_argument("--lora-r",        type=int,   default=16)
    parser.add_argument("--max-examples",  type=int,   default=None,
                        help="Subsample training set for ablation study")
    parser.add_argument("--output-dir",    default=str(OUTPUT_DIR),
                        help="Directory to save adapter (for ablation runs)")
    args = parser.parse_args()
    train(
        base_model=args.base_model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        max_examples=args.max_examples,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
