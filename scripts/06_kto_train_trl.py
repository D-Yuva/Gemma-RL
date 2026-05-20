"""
06_kto_train_trl.py — KTO Training via HuggingFace TRL (Fallback Path)

Use this if mlx native training encounters issues.  TRL's KTOTrainer
runs on PyTorch+MPS on Mac M4 Pro (Metal backend).  It is 2-3× slower
than the native MLX path but is battle-tested and works out of the box.

Prerequisite
------------
  pip install trl peft transformers datasets accelerate

Usage
-----
  python scripts/06_kto_train_trl.py
  python scripts/06_kto_train_trl.py --data data/splits/kto_train.jsonl \\
      --sft-adapter checkpoints/sft --output checkpoints/kto --iters 3000
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


def load_kto_stats() -> dict:
    stats_path = Path(config.DATA_SPLITS) / "kto_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return {"lambda_d": 1.0, "lambda_u": 1.0}


def load_kto_dataset(path: str):
    """
    Load KTO data into HuggingFace Dataset format expected by TRL's KTOTrainer.
    Required columns: prompt, completion, label (bool).
    """
    from datasets import Dataset

    with open(path, "r", encoding="utf-8") as f:
        raw = [json.loads(l) for l in f if l.strip()]

    records = [
        {
            "prompt":     ex["prompt"],
            "completion": ex["completion"],
            "label":      bool(ex["label"]),
        }
        for ex in raw
    ]
    return Dataset.from_list(records)


def main(args):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, PeftModel
        from trl import KTOConfig, KTOTrainer
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install trl peft transformers datasets accelerate")
        sys.exit(1)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # ── Device setup ──────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = "mps"
        print("[device] Mac M4 Pro — using MPS (Metal)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"[device] CUDA GPU — {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[device] CPU — training will be slow")

    # ── Load base model ───────────────────────────────────────────────────────
    print(f"\nLoading base model: {config.BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in bf16 for Mac (no int8 on MPS)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.config.use_cache = False

    # ── Load SFT adapter (if present) ────────────────────────────────────────
    sft_path = Path(args.sft_adapter)
    if (sft_path / "adapter_config.json").exists() or (sft_path / "adapters.npz").exists():
        print(f"Loading SFT adapter from: {args.sft_adapter}")
        try:
            model = PeftModel.from_pretrained(model, args.sft_adapter)
            model = model.merge_and_unload()   # merge SFT weights into base
            print("  ✅ SFT adapter merged into base model")
        except Exception as e:
            print(f"  [warn] Could not load SFT adapter ({e}). Training from base model.")
    else:
        print(f"  [info] No SFT adapter found at {sft_path}. Training from base model.")

    # ── Add new LoRA adapters for KTO phase ──────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading KTO dataset from: {args.data}")
    dataset = load_kto_dataset(args.data)
    n_des = sum(1 for x in dataset if x["label"])
    n_und = len(dataset) - n_des
    print(f"  → {len(dataset)} examples ({n_des} desirable, {n_und} undesirable)")

    # ── Load λ values from stats (or auto-compute) ────────────────────────────
    stats = load_kto_stats()
    lambda_d = args.lambda_d if args.lambda_d > 0 else stats.get("lambda_d", 1.0)
    lambda_u = args.lambda_u if args.lambda_u > 0 else stats.get("lambda_u", 1.0)
    print(f"  → λD={lambda_d:.4f}  λU={lambda_u:.4f}")

    # ── KTO Configuration ─────────────────────────────────────────────────────
    # Steps to train (TRL uses steps, not iters)
    # effective_batch = batch_size × grad_accum = 2 × 8 = 16
    total_steps = args.iters

    kto_config = KTOConfig(
        # Core KTO parameters
        beta=args.beta,
        lambda_desirable=lambda_d,
        lambda_undesirable=lambda_u,
        # Lengths
        max_length=args.max_seq_len,
        max_prompt_length=args.max_seq_len // 2,
        max_completion_length=args.max_seq_len // 2,
        # Optimization
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=args.batch_size,       # min 2 for KL estimate
        gradient_accumulation_steps=args.grad_accum,       # effective batch = 16
        max_steps=total_steps,
        optim="adamw_torch",                               # AdamW (KTO paper default)
        # Precision
        bf16=True,
        fp16=False,
        # Logging & saving
        logging_steps=args.log_every,
        save_steps=args.save_every,
        eval_steps=args.save_every,
        output_dir=args.output,
        report_to="none",
        # Misc
        seed=42,
        remove_unused_columns=False,
        dataloader_num_workers=0,           # 0 for MPS stability
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = KTOTrainer(
        model=model,
        ref_model=None,        # None → TRL creates a frozen copy automatically
        args=kto_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"KTO Training — TRL KTOTrainer")
    print(f"  Model  : {config.BASE_MODEL}")
    print(f"  β      : {args.beta}")
    print(f"  lr     : {args.lr}")
    print(f"  λD     : {lambda_d:.4f}  |  λU : {lambda_u:.4f}")
    print(f"  Steps  : {total_steps}  |  eff. batch: {args.batch_size * args.grad_accum}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    train_result = trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    trainer.save_model(args.output)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    with open(f"logs/kto_trl.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ KTO training (TRL) complete!")
    print(f"   Adapter saved to: {args.output}/")
    print(f"   Metrics saved to: logs/kto_trl.json")
    print(f"\nNext: python scripts/07_kto_raft.py --kto-adapter {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KTO Training via TRL (fallback)")
    parser.add_argument("--data",         default="data/splits/kto_train.jsonl")
    parser.add_argument("--sft-adapter",  default="checkpoints/sft")
    parser.add_argument("--output",       default="checkpoints/kto")
    parser.add_argument("--beta",         type=float, default=config.KTO_BETA)
    parser.add_argument("--lr",           type=float, default=config.KTO_LR)
    parser.add_argument("--lambda-d",     type=float, default=0,   dest="lambda_d")
    parser.add_argument("--lambda-u",     type=float, default=0,   dest="lambda_u")
    parser.add_argument("--batch-size",   type=int,   default=config.KTO_BATCH_SIZE)
    parser.add_argument("--grad-accum",   type=int,   default=config.KTO_GRAD_ACCUM)
    parser.add_argument("--iters",        type=int,   default=config.KTO_ITERS)
    parser.add_argument("--max-seq-len",  type=int,   default=config.KTO_MAX_SEQ_LEN)
    parser.add_argument("--lora-rank",    type=int,   default=config.KTO_LORA_RANK)
    parser.add_argument("--log-every",    type=int,   default=config.KTO_LOG_EVERY)
    parser.add_argument("--save-every",   type=int,   default=config.KTO_SAVE_EVERY)
    args = parser.parse_args()
    main(args)
