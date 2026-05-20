"""
05b_prm_train.py — Process Reward Model (PRM) Training on PRM800K

What it does
------------
Fine-tunes Gemma 4 E4B as a discriminative step-quality classifier using
PRM800K's step-level labels.  The model learns to predict whether each
reasoning step is correct ("positive") or incorrect ("negative").

During RAFT (script 07), the PRM can optionally re-rank or filter generated
solutions step-by-step, beyond the simple outcome-matching ORM.

The PRM is trained as a token-classification / next-token generation task:
  Prompt : "Is the following reasoning step correct?\nStep: {step}\n"
  Label  : "yes" (positive) or "no" (negative)

Inference: PRM score = log P("yes") − log P("no")

This "verbalized" PRM approach avoids adding a regression head and works
natively with mlx_lm.lora's SFT training loop.

Usage
-----
  python scripts/05b_prm_train.py

Optional (skip PRM, use rule-based ORM only):
  In 07_kto_raft.py, pass --use-orm-only to skip PRM scoring.
"""
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config

random.seed(42)


# ─── Dataset Construction ─────────────────────────────────────────────────────

def load_prm800k(cache_dir: str) -> list[dict]:
    """
    Load PRM800K from HuggingFace.  The dataset has 'steps' with labels
    +1 (positive / correct) and -1 (negative / incorrect).

    We convert each step into a binary classification example.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ datasets not installed. Run: pip install datasets")
        sys.exit(1)

    print("Downloading PRM800K (openai/prm800k)...")
    try:
        ds = load_dataset("openai/prm800k", "phase2_train", cache_dir=cache_dir, trust_remote_code=True)
        split = ds["train"]
        print(f"  → Loaded {len(split)} problems from PRM800K phase2_train")
        return list(split)
    except Exception as e:
        print(f"[warn] PRM800K load failed: {e}")
        print("[warn] Trying alternative split name...")
        try:
            ds = load_dataset("openai/prm800k", cache_dir=cache_dir, trust_remote_code=True)
            available = list(ds.keys())
            print(f"  Available splits: {available}")
            split = ds[available[0]]
            return list(split)
        except Exception as e2:
            print(f"❌ Could not load PRM800K: {e2}")
            print("   Falling back to synthetic PRM data generation from GSM8K...")
            return []


def build_prm_examples_from_prm800k(raw_data: list) -> list[dict]:
    """
    Convert PRM800K examples to verbalized binary classification format.
    Expected PRM800K schema: {question, steps: [{completions: [{text, rating}]}]}
    """
    examples = []
    for item in raw_data:
        question = item.get("question", {})
        if isinstance(question, dict):
            question = question.get("problem", "")
        steps_list = item.get("steps", [])

        prev_steps = ""
        for step_data in steps_list:
            completions = step_data.get("completions", [])
            if not completions:
                continue
            # Use the first completion of each step
            comp = completions[0]
            step_text = comp.get("text", "").strip()
            rating = comp.get("rating", 0)  # +1 = positive, -1 = negative, 0 = neutral

            if not step_text or rating == 0:
                continue

            label_word = "yes" if rating > 0 else "no"
            prompt = (
                f"Problem: {question}\n"
                f"Previous steps:{prev_steps if prev_steps else ' (none)'}\n"
                f"Current step: {step_text}\n"
                f"Is this reasoning step correct? Answer yes or no:"
            )
            completion = f" {label_word}"

            examples.append({
                "prompt":     prompt,
                "completion": completion,
                "label":      label_word,
                "source":     "prm800k",
            })
            # Append step to context for the next step
            prev_steps += f"\n{step_text}"

    return examples


def build_synthetic_prm_from_gsm8k(cache_dir: str, n_samples: int = 5000) -> list[dict]:
    """
    Fallback: Build a synthetic PRM dataset from GSM8K.
    Each step in the gold solution is labeled 'yes'.
    We generate wrong steps by corrupting operations (swap + for -, etc.).
    """
    from datasets import load_dataset

    print("[fallback] Building synthetic PRM data from GSM8K...")
    ds = load_dataset("gsm8k", "main", cache_dir=cache_dir, trust_remote_code=True)
    examples = []

    for ex in ds["train"]:
        question = ex["question"]
        answer_raw = ex["answer"]
        parts = answer_raw.split("####")
        steps_text = parts[0].strip()
        steps = [s.strip() for s in steps_text.split("\n") if s.strip()]
        gt_answer = parts[1].strip() if len(parts) > 1 else ""

        prev = ""
        for step in steps:
            if not step:
                continue
            # Real step → label = yes
            prompt = (
                f"Problem: {question}\n"
                f"Previous steps:{prev if prev else ' (none)'}\n"
                f"Current step: {step}\n"
                f"Is this reasoning step correct? Answer yes or no:"
            )
            examples.append({"prompt": prompt, "completion": " yes", "label": "yes", "source": "gsm8k_synthetic"})

            # Corrupt the step → label = no
            corrupted = _corrupt_step(step)
            if corrupted != step:
                examples.append({
                    "prompt":     prompt.replace(step, corrupted),
                    "completion": " no",
                    "label":      "no",
                    "source":     "gsm8k_synthetic_corrupt",
                })
            prev += f"\n{step}"

    random.shuffle(examples)
    examples = examples[:n_samples]
    print(f"  → {len(examples)} synthetic PRM examples")
    return examples


def _corrupt_step(step: str) -> str:
    """Corrupt a reasoning step by swapping arithmetic operators or numbers."""
    swaps = [("+", "-"), ("*", "/"), ("×", "÷")]
    for a, b in swaps:
        if a in step:
            return step.replace(a, b, 1)
    # Swap a number
    nums = re.findall(r"\b\d+\b", step)
    if nums:
        target = random.choice(nums)
        offset = random.choice([-1, 1, 2, -2])
        try:
            wrong = str(int(target) + offset)
            return step.replace(target, wrong, 1)
        except ValueError:
            pass
    return step


def save_prm_splits(examples: list[dict], splits_dir: str):
    """Save train/valid splits as JSONL files for mlx_lm.lora training."""
    Path(splits_dir).mkdir(parents=True, exist_ok=True)
    random.shuffle(examples)
    val_n = min(500, max(1, len(examples) // 10))
    train_ex = examples[:-val_n]
    val_ex   = examples[-val_n:]

    # mlx_lm.lora format: each line has "text" = prompt + completion
    def to_mlx_format(ex: dict) -> dict:
        return {"text": ex["prompt"] + ex["completion"]}

    train_path = Path(splits_dir) / "prm_train.jsonl"
    val_path   = Path(splits_dir) / "prm_valid.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_ex:
            f.write(json.dumps(to_mlx_format(ex), ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_ex:
            f.write(json.dumps(to_mlx_format(ex), ensure_ascii=False) + "\n")

    print(f"\n  PRM train : {len(train_ex)} examples → {train_path}")
    print(f"  PRM valid : {len(val_ex)} examples → {val_path}")
    return str(train_path), str(val_path)


# ─── MLX Training Command Generator ──────────────────────────────────────────

def print_training_command(splits_dir: str):
    """Print the mlx_lm.lora command to run PRM fine-tuning."""
    print("\n" + "=" * 60)
    print("PRM TRAINING COMMAND (run this in your terminal):")
    print("=" * 60)
    cmd = f"""
python -m mlx_lm.lora \\
    --model {config.BASE_MODEL} \\
    --adapter-path checkpoints/sft \\
    --train \\
    --data {splits_dir} \\
    --train-split prm_train \\
    --valid-split prm_valid \\
    --batch-size {config.PRM_BATCH_SIZE} \\
    --iters {config.PRM_ITERS} \\
    --learning-rate {config.PRM_LR} \\
    --lora-rank {config.PRM_LORA_RANK} \\
    --lora-scale {config.PRM_LORA_RANK * 2}.0 \\
    --num-layers 8 \\
    --max-seq-length {config.PRM_MAX_SEQ_LEN} \\
    --save-every {config.PRM_SAVE_EVERY} \\
    --adapter-path checkpoints/prm \\
    --val-batches 25 \\
    --steps-per-eval 250 \\
    2>&1 | tee logs/prm.log
"""
    print(cmd)


# ─── PRM Inference Helper (used by 07_kto_raft.py) ──────────────────────────

def prm_score(model, tokenizer, prompt: str, step: str, prev_steps: str = "") -> float:
    """
    Returns a scalar PRM score for a reasoning step.
    score > 0  → step likely correct
    score < 0  → step likely incorrect
    Uses log P("yes") - log P("no") difference.
    """
    import mlx.core as mx
    import mlx.nn as nn

    prm_prompt = (
        f"Problem: {prompt}\n"
        f"Previous steps:{prev_steps if prev_steps else ' (none)'}\n"
        f"Current step: {step}\n"
        f"Is this reasoning step correct? Answer yes or no:"
    )
    input_ids = tokenizer.encode(prm_prompt, return_tensors=None)
    input_arr = mx.array([input_ids])

    logits = model(input_arr)          # (1, T, vocab_size)
    last_logit = logits[0, -1, :]      # (vocab_size,) — predict next token

    log_probs = nn.log_softmax(last_logit, axis=-1)

    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode("no",  add_special_tokens=False)[0]

    score = log_probs[yes_id].item() - log_probs[no_id].item()
    return score


def solution_prm_score(model, tokenizer, problem: str, solution: str) -> float:
    """
    Compute the minimum step PRM score across all steps in a solution.
    This is the "process reward" — a single wrong step brings the score down.
    """
    # Extract steps from <think>...</think>
    m = re.search(r"<think>(.*?)</think>", solution, re.DOTALL)
    if not m:
        return 0.0

    steps_text = m.group(1).strip()
    steps = [s.strip() for s in steps_text.split("\n") if s.strip()]

    if not steps:
        return 0.0

    scores = []
    prev = ""
    for step in steps:
        s = prm_score(model, tokenizer, problem, step, prev)
        scores.append(s)
        prev += f"\n{step}"

    return min(scores)   # worst step determines overall correctness


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 5b: Process Reward Model (PRM) Data Preparation")
    print("=" * 60)

    Path("logs").mkdir(exist_ok=True)
    Path(config.DATA_SPLITS).mkdir(parents=True, exist_ok=True)

    # 1. Try PRM800K
    raw = load_prm800k(config.DATA_RAW)

    if raw:
        print(f"\nBuilding PRM examples from {len(raw)} PRM800K entries...")
        examples = build_prm_examples_from_prm800k(raw)
        print(f"  → {len(examples)} step-level examples")
        if len(examples) < 1000:
            print("[warn] Few examples from PRM800K. Supplementing with GSM8K synthetic...")
            extra = build_synthetic_prm_from_gsm8k(config.DATA_RAW, n_samples=3000)
            examples += extra
    else:
        examples = build_synthetic_prm_from_gsm8k(config.DATA_RAW, n_samples=8000)

    # Balance yes/no
    yes_ex = [e for e in examples if e["label"] == "yes"]
    no_ex  = [e for e in examples if e["label"] == "no"]
    n = min(len(yes_ex), len(no_ex))
    balanced = random.sample(yes_ex, n) + random.sample(no_ex, n)
    print(f"\nBalanced PRM dataset: {n} positive + {n} negative = {len(balanced)} total")

    # 2. Save splits
    _, _ = save_prm_splits(balanced, config.DATA_SPLITS)

    # 3. Print training command
    print_training_command(config.DATA_SPLITS)

    print("\n✅ PRM data prep complete.")
    print("   Run the command above to train the PRM.")
    print("   Checkpoint will be saved to: checkpoints/prm/")
    print("\nNote: PRM is OPTIONAL. Script 07_kto_raft.py uses rule-based ORM by default.")
    print("      Pass --prm-adapter checkpoints/prm to 07_kto_raft.py to enable it.")


if __name__ == "__main__":
    main()
