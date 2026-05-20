"""
07_kto_raft.py — KTO-RAFT Self-Improvement Loop (3 Iterations)

Algorithm per iteration
-----------------------
  1. GENERATE  : Sample K=4 solutions per problem using current best model
  2. ORM FILTER: Check final answer correctness (rule-based + optional PRM)
  3. LABEL     : correct → desirable (True), incorrect → undesirable (False)
  4. BALANCE   : Compute λD, λU dynamically from this iteration's ratio
  5. KTO UPDATE: Run KTO training on labeled outputs
  6. EVALUATE  : GSM8K accuracy on 200 held-out problems
  7. CHECKPOINT: Save best adapter (highest GSM8K)

Expected GSM8K progression (based on KTO paper + Qwen3B proxy):
  SFT baseline     : ~39%
  KTO phase (step 6): ~50-54%  ← target ≥50%
  RAFT iter 1      : ~56%
  RAFT iter 2      : ~58%
  RAFT iter 3      : ~60%      ← ceiling with 4B model

Usage
-----
  # Run all 3 RAFT iterations (starting from KTO checkpoint):
  python scripts/07_kto_raft.py --kto-adapter checkpoints/kto

  # Run a single iteration (e.g. iteration 2):
  python scripts/07_kto_raft.py --kto-adapter checkpoints/kto-raft/iter1 --start-iter 2

  # Use PRM for step-level scoring (optional):
  python scripts/07_kto_raft.py --kto-adapter checkpoints/kto --prm-adapter checkpoints/prm

  # Use rule-based ORM only (default, fastest):
  python scripts/07_kto_raft.py --kto-adapter checkpoints/kto --use-orm-only
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config

random.seed(42)


# ─── Answer Extraction & ORM ──────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    """Extract the final answer from model output."""
    # Try <answer>...</answer> tag first
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip().replace(",", "")
    # Fallback: last number
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


def answers_match(pred: str, gt: str) -> bool:
    """Fuzzy numeric equality (exact string or ±0.01 float tolerance)."""
    pred = pred.strip().lower().replace(",", "").rstrip(".")
    gt   = gt.strip().lower().replace(",", "").rstrip(".")
    if pred == gt:
        return True
    try:
        return abs(float(pred) - float(gt)) < 0.01
    except ValueError:
        return False


def orm_check(response: str, gt_answer: str) -> bool:
    """Outcome Reward Model: check if predicted answer matches ground truth."""
    pred = extract_answer(response)
    return answers_match(pred, gt_answer)


# ─── Solution Generation ──────────────────────────────────────────────────────

def generate_solutions(
    model,
    tokenizer,
    problems: list[dict],
    k: int,
    temp: float,
    max_tokens: int,
) -> list[dict]:
    """
    For each problem, generate K solutions.
    Returns flat list of {prompt, completion, gt_answer} dicts.
    """
    from mlx_lm import generate as mlx_generate

    all_samples = []
    n = len(problems)

    for i, prob in enumerate(problems):
        prompt    = prob["prompt"]
        gt_answer = prob.get("answer", "")

        for attempt in range(k):
            try:
                response = mlx_generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temp=temp,
                    verbose=False,
                )
                all_samples.append({
                    "prompt":     prompt,
                    "completion": response,
                    "gt_answer":  gt_answer,
                    "source":     prob.get("source", "unknown"),
                    "task":       prob.get("task", "unknown"),
                })
            except Exception as e:
                print(f"  [warn] Generation error (problem {i}, attempt {attempt}): {e}")

        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  [{i+1:4d}/{n}] problems sampled  ({len(all_samples)} total completions)", end="\r")

    print()
    return all_samples


# ─── Optional PRM Scoring ─────────────────────────────────────────────────────

def load_prm_model(prm_adapter_path: str):
    """Load PRM model for step-level scoring (optional)."""
    from mlx_lm import load
    try:
        print(f"[prm] Loading PRM adapter from {prm_adapter_path}...")
        prm_model, prm_tok = load(config.BASE_MODEL, adapter_path=prm_adapter_path)
        print(f"[prm] PRM model loaded.")
        return prm_model, prm_tok
    except Exception as e:
        print(f"[prm] WARNING: Could not load PRM model ({e}). Falling back to ORM only.")
        return None, None


def score_with_prm(prm_model, prm_tokenizer, prompt: str, completion: str) -> float:
    """Compute PRM score for a solution (min step score across all steps)."""
    # Import from 05b_prm_train.py helper
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from scripts.prm_train import solution_prm_score
    except ImportError:
        try:
            from prm_train import solution_prm_score
        except ImportError:
            # Inline the solution_prm_score logic
            import mlx.core as mx
            import mlx.nn as nn

            m = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
            if not m:
                return 0.0
            steps = [s.strip() for s in m.group(1).split("\n") if s.strip()]
            if not steps:
                return 0.0

            scores = []
            prev = ""
            for step in steps:
                prm_prompt = (
                    f"Problem: {prompt}\n"
                    f"Previous steps:{prev if prev else ' (none)'}\n"
                    f"Current step: {step}\n"
                    f"Is this reasoning step correct? Answer yes or no:"
                )
                ids = prm_tokenizer.encode(prm_prompt, return_tensors=None)
                arr = mx.array([ids])
                out = prm_model(arr)
                logits = out[0] if isinstance(out, tuple) else out
                last = logits[0, -1, :]
                lp = nn.log_softmax(last, axis=-1)
                yes_id = prm_tokenizer.encode("yes", add_special_tokens=False)[0]
                no_id  = prm_tokenizer.encode("no",  add_special_tokens=False)[0]
                scores.append(float(lp[yes_id].item()) - float(lp[no_id].item()))
                prev += f"\n{step}"
            return min(scores) if scores else 0.0

    return solution_prm_score(prm_model, prm_tokenizer, prompt, completion)


# ─── KTO Label Assignment ─────────────────────────────────────────────────────

def label_samples(
    samples: list[dict],
    prm_model=None,
    prm_tokenizer=None,
    prm_threshold: float = 0.0,
) -> list[dict]:
    """
    Apply ORM (and optionally PRM) to assign binary labels.
    Returns KTO-formatted examples with 'label' field.
    """
    labeled = []
    n_correct = 0
    n_wrong   = 0

    for s in samples:
        orm_correct = orm_check(s["completion"], s["gt_answer"])

        if prm_model is not None and orm_correct:
            # Re-score with PRM: even ORM-correct solutions may have flawed reasoning
            prm_s = score_with_prm(prm_model, prm_tokenizer, s["prompt"], s["completion"])
            label = bool(orm_correct) and (prm_s >= prm_threshold)
        else:
            label = orm_correct

        labeled.append({
            "prompt":     s["prompt"],
            "completion": s["completion"],
            "label":      label,
            "source":     s.get("source", "raft"),
            "task":       s.get("task", "math"),
        })
        if label:
            n_correct += 1
        else:
            n_wrong += 1

    return labeled, n_correct, n_wrong


# ─── λD / λU Calculation ──────────────────────────────────────────────────────

def compute_lambda(n_desirable: int, n_undesirable: int) -> tuple[float, float]:
    """
    Compute λD, λU such that λD·nD / (λU·nU) ∈ [1.0, 1.5].
    From KTO paper equation (9).
    """
    if n_desirable == 0 or n_undesirable == 0:
        return 1.0, 1.0
    lambda_u = 1.0
    lambda_d = round(1.5 * n_undesirable / n_desirable, 4)
    ratio    = lambda_d * n_desirable / (lambda_u * n_undesirable)
    return lambda_d, lambda_u


# ─── GSM8K Evaluation ────────────────────────────────────────────────────────

def evaluate_gsm8k(model, tokenizer, n_samples: int = 200) -> dict:
    """
    Quick GSM8K evaluation using greedy decoding.
    Returns {accuracy, n_correct, n_total, samples}.
    """
    from datasets import load_dataset
    from mlx_lm import generate as mlx_generate

    print(f"\n[eval] Evaluating on {n_samples} GSM8K test examples (greedy)...")
    ds    = load_dataset("gsm8k", "main", cache_dir=config.DATA_RAW)["test"]
    items = list(ds)
    random.shuffle(items)
    items = items[:n_samples]

    correct = 0
    results = []

    for i, ex in enumerate(items):
        prompt    = f"Solve step-by-step:\n{ex['question']}"
        parts     = ex["answer"].split("####")
        gt_answer = parts[1].strip() if len(parts) > 1 else ""

        try:
            response = mlx_generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=config.EVAL_MAX_TOKENS,
                temp=0.0,          # greedy
                verbose=False,
            )
            pred = extract_answer(response)
            is_correct = answers_match(pred, gt_answer)
            correct += is_correct
            results.append({"question": ex["question"], "gt": gt_answer,
                            "pred": pred, "correct": is_correct})
        except Exception as e:
            print(f"  [warn] eval error on item {i}: {e}")
            results.append({"question": ex["question"], "gt": gt_answer,
                            "pred": "", "correct": False})

        if (i + 1) % 50 == 0:
            acc = 100 * correct / (i + 1)
            print(f"  [{i+1}/{n_samples}] running accuracy: {acc:.1f}%", end="\r")

    accuracy = 100 * correct / max(len(items), 1)
    print(f"\n[eval] GSM8K accuracy: {accuracy:.2f}%  ({correct}/{len(items)})")
    return {"accuracy": accuracy, "n_correct": correct, "n_total": len(items), "samples": results[:10]}


# ─── KTO Training Invocation ──────────────────────────────────────────────────

def run_kto_training(
    data_path: str,
    adapter_path: str,
    output_dir: str,
    lambda_d: float,
    lambda_u: float,
    iters: int,
    use_trl: bool = False,
) -> bool:
    """
    Invoke KTO training as a subprocess (MLX or TRL).
    Returns True if training exited successfully.
    """
    script = "06_kto_train_trl.py" if use_trl else "06b_kto_train_mlx.py"
    script_path = str(Path(__file__).parent / script)

    cmd = [
        sys.executable, script_path,
        "--data",        data_path,
        "--sft-adapter", adapter_path,
        "--output",      output_dir,
        "--iters",       str(iters),
        "--lambda-d",    str(lambda_d),
        "--lambda-u",    str(lambda_u),
    ]

    print(f"\n[raft] Running KTO update: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


# ─── Main RAFT Loop ───────────────────────────────────────────────────────────

def main(args):
    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints/kto-raft").mkdir(parents=True, exist_ok=True)

    # ── Load GSM8K training problems for generation ───────────────────────────
    print("Loading GSM8K training problems for RAFT generation...")
    from datasets import load_dataset
    gsm_train = list(load_dataset("gsm8k", "main", cache_dir=config.DATA_RAW)["train"])
    problems  = [
        {
            "prompt":  f"Solve step-by-step:\n{ex['question']}",
            "answer":  ex["answer"].split("####")[1].strip() if "####" in ex["answer"] else "",
            "source":  "gsm8k",
            "task":    "math",
        }
        for ex in gsm_train
    ]
    print(f"  → {len(problems)} GSM8K training problems available")

    # ── Load PRM (optional) ───────────────────────────────────────────────────
    prm_model, prm_tokenizer = None, None
    if args.prm_adapter and not args.use_orm_only:
        prm_model, prm_tokenizer = load_prm_model(args.prm_adapter)

    # ── RAFT iteration tracking ───────────────────────────────────────────────
    raft_log    = []
    best_acc    = -1.0
    best_ckpt   = args.kto_adapter       # start from KTO checkpoint
    current_ckpt = args.kto_adapter

    for raft_iter in range(args.start_iter, args.start_iter + config.RAFT_N_ITERS):
        print(f"\n{'='*60}")
        print(f"  KTO-RAFT Iteration {raft_iter} of {args.start_iter + config.RAFT_N_ITERS - 1}")
        print(f"  Current checkpoint: {current_ckpt}")
        print(f"{'='*60}")

        iter_start = time.time()

        # ── Step 1: Load generation model ────────────────────────────────────
        from mlx_lm import load as mlx_load
        print(f"\n[raft {raft_iter}] Loading generation model...")
        gen_model, gen_tok = mlx_load(config.BASE_MODEL, adapter_path=current_ckpt)

        # ── Step 2: Sample problems ───────────────────────────────────────────
        n = min(args.n_problems, len(problems))
        sampled = random.sample(problems, n)
        print(f"[raft {raft_iter}] Generating {config.RAFT_K} solutions × {n} problems = {n * config.RAFT_K} total...")

        raw_samples = generate_solutions(
            gen_model, gen_tok,
            sampled,
            k=config.RAFT_K,
            temp=config.RAFT_TEMP,
            max_tokens=config.RAFT_MAX_GEN_TOKENS,
        )

        # Free generation model from memory
        del gen_model
        mx_eval_barrier = True
        try:
            import mlx.core as mx
            mx.eval(mx.array([0]))   # force eval/drain the computation graph
        except Exception:
            pass

        print(f"[raft {raft_iter}] Generated {len(raw_samples)} raw completions")

        # ── Step 3: Label with ORM (+ optional PRM) ───────────────────────────
        labeled, n_correct, n_wrong = label_samples(
            raw_samples, prm_model, prm_tokenizer, args.prm_threshold
        )
        print(f"[raft {raft_iter}] Labels: {n_correct} desirable, {n_wrong} undesirable  "
              f"(correct rate: {100*n_correct/max(len(labeled),1):.1f}%)")

        if n_correct == 0 or n_wrong == 0:
            print(f"[raft {raft_iter}] WARNING: All examples have the same label. "
                  f"KTO requires both. Skipping this iteration.")
            raft_log.append({"iter": raft_iter, "skipped": True, "n_correct": n_correct, "n_wrong": n_wrong})
            continue

        # ── Step 4: Compute λD, λU for this iteration ─────────────────────────
        lambda_d, lambda_u = compute_lambda(n_correct, n_wrong)
        ratio = lambda_d * n_correct / (lambda_u * n_wrong)
        print(f"[raft {raft_iter}] λD={lambda_d:.4f}  λU={lambda_u:.4f}  "
              f"(λD·nD/λU·nU = {ratio:.3f}  ← target [1.0, 1.5])")

        # ── Step 5: Save this iteration's KTO data ────────────────────────────
        data_path = f"data/processed/raft_iter{raft_iter}_kto.jsonl"
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, "w", encoding="utf-8") as f:
            for ex in labeled:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[raft {raft_iter}] Saved {len(labeled)} examples → {data_path}")

        # ── Step 6: KTO update ────────────────────────────────────────────────
        out_ckpt = f"checkpoints/kto-raft/iter{raft_iter}"
        success = run_kto_training(
            data_path    = data_path,
            adapter_path = current_ckpt,
            output_dir   = out_ckpt,
            lambda_d     = lambda_d,
            lambda_u     = lambda_u,
            iters        = config.RAFT_KTO_ITERS,
            use_trl      = args.use_trl,
        )

        if not success:
            print(f"[raft {raft_iter}] WARNING: KTO training failed (non-zero exit). "
                  f"Keeping previous checkpoint.")
            raft_log.append({"iter": raft_iter, "kto_failed": True})
            continue

        # ── Step 7: Evaluate ──────────────────────────────────────────────────
        eval_model, eval_tok = mlx_load(config.BASE_MODEL, adapter_path=out_ckpt)
        eval_results = evaluate_gsm8k(eval_model, eval_tok, n_samples=args.eval_samples)
        del eval_model

        acc = eval_results["accuracy"]

        # ── Track best ────────────────────────────────────────────────────────
        if acc > best_acc:
            best_acc  = acc
            best_ckpt = out_ckpt
            print(f"  🏆 New best! GSM8K accuracy: {acc:.2f}% → {out_ckpt}")
        else:
            print(f"  GSM8K accuracy: {acc:.2f}% (best so far: {best_acc:.2f}%)")

        # Update current checkpoint for next iteration
        current_ckpt = out_ckpt

        iter_elapsed = time.time() - iter_start

        # ── Save iteration log ────────────────────────────────────────────────
        iter_log = {
            "iter":         raft_iter,
            "n_problems":   n,
            "n_generated":  len(raw_samples),
            "n_desirable":  n_correct,
            "n_undesirable": n_wrong,
            "correct_rate": round(100 * n_correct / max(len(labeled), 1), 2),
            "lambda_d":     lambda_d,
            "lambda_u":     lambda_u,
            "kto_iters":    config.RAFT_KTO_ITERS,
            "gsm8k_acc":    round(acc, 3),
            "is_best":      acc >= best_acc,
            "elapsed_min":  round(iter_elapsed / 60, 1),
            "checkpoint":   out_ckpt,
        }
        raft_log.append(iter_log)

        log_path = "logs/raft_log.json"
        with open(log_path, "w") as f:
            json.dump(raft_log, f, indent=2)
        print(f"[raft {raft_iter}] Iteration log saved → {log_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"KTO-RAFT Complete — {config.RAFT_N_ITERS} iterations")
    print(f"{'='*60}")
    print(f"\n  Best GSM8K accuracy : {best_acc:.2f}%")
    print(f"  Best checkpoint     : {best_ckpt}")
    print(f"\n  Per-iteration GSM8K accuracy:")
    for entry in raft_log:
        if "gsm8k_acc" in entry:
            marker = " 🏆" if entry.get("is_best") else ""
            print(f"    Iter {entry['iter']}: {entry['gsm8k_acc']:.2f}%{marker}")

    print(f"\nNext: python scripts/08_evaluate.py --adapter {best_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KTO-RAFT Self-Improvement Loop")
    parser.add_argument("--kto-adapter",   default="checkpoints/kto",
                        dest="kto_adapter",
                        help="Starting KTO checkpoint from 06b_kto_train_mlx.py")
    parser.add_argument("--prm-adapter",   default="",  dest="prm_adapter",
                        help="PRM adapter path (optional). Empty = ORM only.")
    parser.add_argument("--use-orm-only",  action="store_true", dest="use_orm_only",
                        help="Use rule-based ORM only, skip PRM scoring.")
    parser.add_argument("--prm-threshold", type=float, default=0.0, dest="prm_threshold",
                        help="Min PRM score to count as desirable (default: 0.0 = any positive).")
    parser.add_argument("--n-problems",    type=int, default=config.RAFT_N_PROBLEMS,
                        dest="n_problems",
                        help="Problems sampled per RAFT iteration.")
    parser.add_argument("--eval-samples",  type=int, default=200, dest="eval_samples",
                        help="GSM8K test samples for per-iteration eval.")
    parser.add_argument("--start-iter",   type=int, default=1,   dest="start_iter",
                        help="Which iteration to start from (useful for resuming).")
    parser.add_argument("--use-trl",      action="store_true",   dest="use_trl",
                        help="Use TRL KTO trainer instead of MLX native.")
    args = parser.parse_args()
    main(args)
