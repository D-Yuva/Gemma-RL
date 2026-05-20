"""
08_evaluate.py — Unified Benchmark Evaluation Suite

Evaluates the fine-tuned model across three benchmarks:
  1. GSM8K      : Grade-school math (primary target: ≥50%)
  2. MMLU       : Massive Multitask Language Understanding (target: ≥45%)
  3. StrategyQA : Multi-hop yes/no reasoning (proxy for BBH, target: ≥60%)

All three benchmarks use greedy decoding (temp=0.0) for reproducibility.

Baseline comparisons are printed alongside results:
  SFT baseline   : GSM8K 39.0%, MMLU 57.0%, StrategyQA ~55%  (from KTO paper)
  DPO alignment  : GSM8K 40.0%, MMLU 58.2%, StrategyQA ~54%
  KTO alignment  : GSM8K 53.5%, MMLU 58.6%, StrategyQA ~63%  ← paper target

Usage
-----
  # Evaluate best RAFT checkpoint:
  python scripts/08_evaluate.py --adapter checkpoints/kto-raft/iter3

  # Evaluate SFT baseline (no KTO):
  python scripts/08_evaluate.py --adapter checkpoints/sft --tag sft_baseline

  # Evaluate all stages and compare:
  python scripts/08_evaluate.py --all-stages

  # Quick smoke-test (50 samples per benchmark):
  python scripts/08_evaluate.py --adapter checkpoints/kto --quick
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config

random.seed(42)


# ─── Baselines (from KTO paper Table 2) ──────────────────────────────────────

BASELINES = {
    "SFT (no alignment)":    {"gsm8k": 39.0, "mmlu": 57.0, "strategy_qa": 55.0},
    "DPO (current SOTA)":    {"gsm8k": 40.0, "mmlu": 58.2, "strategy_qa": 54.0},
    "KTO β=0.1 (paper)":     {"gsm8k": 53.5, "mmlu": 58.6, "strategy_qa": 63.0},
    "Random baseline":        {"gsm8k":  0.0, "mmlu": 25.0, "strategy_qa": 50.0},
}

KPI_TARGETS = {
    "gsm8k":       50.0,
    "mmlu":        45.0,
    "strategy_qa": 60.0,
}


# ─── Answer Extraction ────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


def answers_match(pred: str, gt: str) -> bool:
    pred = pred.strip().lower().replace(",", "").rstrip(".")
    gt   = gt.strip().lower().replace(",", "").rstrip(".")
    if pred == gt:
        return True
    try:
        return abs(float(pred) - float(gt)) < 0.01
    except ValueError:
        return False


def extract_yesno(text: str) -> str:
    """Extract yes/no from model output."""
    text_lower = text.lower().strip()
    # Check <answer> tag
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text_lower, re.DOTALL)
    if m:
        ans = m.group(1).strip()
        if "yes" in ans:
            return "yes"
        if "no" in ans:
            return "no"
    # Direct match
    for word in ["yes", "no"]:
        if text_lower.startswith(word) or f"\n{word}" in text_lower:
            return word
    return ""


def extract_mcq_letter(text: str) -> str:
    """Extract MMLU answer letter (A/B/C/D) from model output."""
    text = text.strip().upper()
    # Check <answer> tag
    m = re.search(r"<answer>\s*([ABCD])\s*</answer>", text, re.DOTALL)
    if m:
        return m.group(1)
    # First letter match
    for letter in "ABCD":
        if text.startswith(letter) or f"\n{letter}" in text or f"({letter})" in text:
            return letter
    return ""


# ─── GSM8K Evaluation ─────────────────────────────────────────────────────────

def evaluate_gsm8k(model, tokenizer, n_samples: int, verbose: bool = False) -> dict:
    from datasets import load_dataset
    from mlx_lm import generate as mlx_generate

    print(f"\n{'─'*50}")
    print(f"  Benchmark: GSM8K ({n_samples} samples)")
    print(f"{'─'*50}")

    ds    = load_dataset("gsm8k", "main", cache_dir=config.DATA_RAW)["test"]
    items = list(ds)
    random.shuffle(items)
    items = items[:n_samples]

    correct    = 0
    all_preds  = []
    latencies  = []
    t0 = time.time()

    for i, ex in enumerate(items):
        prompt    = f"Solve step-by-step:\n{ex['question']}"
        gt_answer = ex["answer"].split("####")[1].strip() if "####" in ex["answer"] else ""

        t_start = time.time()
        try:
            response = mlx_generate(model, tokenizer, prompt=prompt,
                                    max_tokens=config.EVAL_MAX_TOKENS, temp=0.0, verbose=False)
            latencies.append(time.time() - t_start)
            pred = extract_answer(response)
            is_correct = answers_match(pred, gt_answer)
            correct += is_correct
            all_preds.append({"gt": gt_answer, "pred": pred, "correct": is_correct})
        except Exception as e:
            if verbose:
                print(f"  [warn] Error on item {i}: {e}")
            all_preds.append({"gt": gt_answer, "pred": "", "correct": False})
            latencies.append(0)

        if (i + 1) % 100 == 0:
            acc = 100 * correct / (i + 1)
            print(f"  [{i+1:4d}/{n_samples}] running accuracy: {acc:.1f}%", end="\r")

    total_time = time.time() - t0
    accuracy = 100 * correct / max(len(items), 1)
    avg_lat  = sum(latencies) / max(len(latencies), 1)

    print(f"\n  ✅ GSM8K Accuracy  : {accuracy:.2f}%  ({correct}/{len(items)})")
    print(f"     Avg latency     : {avg_lat:.2f}s/example")
    print(f"     Total time      : {total_time/60:.1f} min")

    return {
        "accuracy":    round(accuracy, 3),
        "n_correct":   correct,
        "n_total":     len(items),
        "avg_lat_s":   round(avg_lat, 3),
        "predictions": all_preds[:20],   # save first 20 for inspection
    }


# ─── MMLU Evaluation ──────────────────────────────────────────────────────────

def evaluate_mmlu(model, tokenizer, n_per_subject: int = 20, verbose: bool = False) -> dict:
    from datasets import load_dataset
    from mlx_lm import generate as mlx_generate

    print(f"\n{'─'*50}")
    print(f"  Benchmark: MMLU ({n_per_subject} samples/subject)")
    print(f"{'─'*50}")

    try:
        ds = load_dataset("cais/mmlu", "all", cache_dir=config.DATA_RAW, trust_remote_code=True)
        test_split = ds["test"]
    except Exception as e:
        print(f"  [warn] Could not load MMLU: {e}")
        print("  Trying alternative MMLU loader...")
        try:
            ds = load_dataset("lukaemon/mmlu", cache_dir=config.DATA_RAW, trust_remote_code=True)
            test_split = ds["test"]
        except Exception as e2:
            print(f"  ❌ MMLU not available: {e2}")
            return {"accuracy": 0.0, "n_correct": 0, "n_total": 0, "error": str(e2)}

    # Organize by subject
    by_subject = {}
    for ex in test_split:
        subj = ex.get("subject", "unknown")
        by_subject.setdefault(subj, []).append(ex)

    choices_letters = ["A", "B", "C", "D"]
    correct   = 0
    total     = 0
    per_subj  = {}

    for subject, items in by_subject.items():
        sample = random.sample(items, min(n_per_subject, len(items)))
        subj_correct = 0

        for ex in sample:
            question = ex.get("question", ex.get("input", ""))
            choices  = ex.get("choices", ex.get("options", []))
            answer   = ex.get("answer", ex.get("target", ""))
            # Normalize answer to letter
            if isinstance(answer, int):
                gt_letter = choices_letters[answer]
            else:
                gt_letter = str(answer).strip().upper()
                if gt_letter not in "ABCD":
                    gt_letter = choices_letters[0]   # default

            # Build MMLU prompt (5-shot style is too long; use 0-shot)
            choices_text = "\n".join(
                f"{choices_letters[j]}. {c}" for j, c in enumerate(choices[:4])
            )
            prompt = (
                f"Question: {question}\n"
                f"{choices_text}\n"
                f"Answer: Think step by step, then put your final answer letter in <answer></answer>."
            )

            try:
                response = mlx_generate(model, tokenizer, prompt=prompt,
                                        max_tokens=256, temp=0.0, verbose=False)
                pred = extract_mcq_letter(response)
                is_correct = pred == gt_letter
                correct    += is_correct
                subj_correct += is_correct
            except Exception as e:
                if verbose:
                    print(f"  [warn] MMLU error ({subject}): {e}")
            total += 1

        subj_acc = 100 * subj_correct / max(len(sample), 1)
        per_subj[subject] = round(subj_acc, 1)

    accuracy = 100 * correct / max(total, 1)
    print(f"  ✅ MMLU Accuracy   : {accuracy:.2f}%  ({correct}/{total}, {len(per_subj)} subjects)")

    return {
        "accuracy":      round(accuracy, 3),
        "n_correct":     correct,
        "n_total":       total,
        "n_subjects":    len(per_subj),
        "per_subject":   per_subj,
    }


# ─── StrategyQA Evaluation ────────────────────────────────────────────────────

def evaluate_strategyqa(model, tokenizer, n_samples: int = 500, verbose: bool = False) -> dict:
    from datasets import load_dataset
    from mlx_lm import generate as mlx_generate

    print(f"\n{'─'*50}")
    print(f"  Benchmark: StrategyQA ({n_samples} samples)")
    print(f"{'─'*50}")

    try:
        ds = load_dataset("ChilleD/StrategyQA", cache_dir=config.DATA_RAW, trust_remote_code=True)
        items = list(ds.get("test", ds.get("validation", ds["train"])))
    except Exception as e:
        print(f"  [warn] Could not load StrategyQA: {e}")
        try:
            ds = load_dataset("wics/strategy-qa", cache_dir=config.DATA_RAW, trust_remote_code=True)
            items = list(ds["test"])
        except Exception as e2:
            print(f"  ❌ StrategyQA not available: {e2}")
            return {"accuracy": 0.0, "n_correct": 0, "n_total": 0, "error": str(e2)}

    random.shuffle(items)
    items = items[:n_samples]

    correct = 0
    total   = 0

    for i, ex in enumerate(items):
        question  = ex.get("question", ex.get("input", ""))
        gt_answer = ex.get("answer", ex.get("target", False))
        # Normalize gt to yes/no string
        if isinstance(gt_answer, bool):
            gt_str = "yes" if gt_answer else "no"
        else:
            gt_str = str(gt_answer).strip().lower()
            if gt_str not in ("yes", "no"):
                gt_str = "yes" if gt_str in ("true", "1") else "no"

        prompt = (
            f"Answer the following yes/no question. Think step by step, "
            f"then put your final answer (yes or no) in <answer></answer>.\n"
            f"Question: {question}"
        )

        try:
            response = mlx_generate(model, tokenizer, prompt=prompt,
                                    max_tokens=256, temp=0.0, verbose=False)
            pred = extract_yesno(response)
            is_correct = pred == gt_str
            correct    += is_correct
            total      += 1
        except Exception as e:
            if verbose:
                print(f"  [warn] StrategyQA error on item {i}: {e}")
            total += 1

    accuracy = 100 * correct / max(total, 1)
    print(f"  ✅ StrategyQA Acc  : {accuracy:.2f}%  ({correct}/{total})")

    return {"accuracy": round(accuracy, 3), "n_correct": correct, "n_total": total}


# ─── Results Printer ──────────────────────────────────────────────────────────

def print_comparison_table(tag: str, results: dict):
    """Print a formatted comparison table against baselines."""
    model_acc = {
        "gsm8k":       results.get("gsm8k",       {}).get("accuracy", 0),
        "mmlu":        results.get("mmlu",         {}).get("accuracy", 0),
        "strategy_qa": results.get("strategy_qa",  {}).get("accuracy", 0),
    }

    print(f"\n{'='*70}")
    print(f"  BENCHMARK RESULTS: {tag}")
    print(f"{'='*70}")
    print(f"  {'Model':<30} {'GSM8K':>8} {'MMLU':>8} {'StratQA':>8}")
    print(f"  {'-'*58}")

    for name, baseline in BASELINES.items():
        print(f"  {name:<30} {baseline['gsm8k']:>7.1f}% {baseline['mmlu']:>7.1f}% {baseline['strategy_qa']:>7.1f}%")

    print(f"  {'-'*58}")
    print(f"  {'Your Model: ' + tag:<30} {model_acc['gsm8k']:>7.1f}% {model_acc['mmlu']:>7.1f}% {model_acc['strategy_qa']:>7.1f}%")
    print(f"{'='*70}")

    print(f"\n  KPI Status:")
    for bench, target in KPI_TARGETS.items():
        achieved = model_acc.get(bench, 0)
        status   = "✅ PASS" if achieved >= target else "❌ MISS"
        gap      = achieved - target
        gap_str  = f"(+{gap:.1f}%)" if gap >= 0 else f"({gap:.1f}%)"
        print(f"    {bench:<15} target≥{target:.0f}%  achieved={achieved:.2f}%  {status}  {gap_str}")
    print()


# ─── Multi-stage Comparison ───────────────────────────────────────────────────

def evaluate_all_stages(quick: bool, verbose: bool):
    """Evaluate all pipeline checkpoints and compare."""
    stages = [
        ("SFT baseline",    "checkpoints/sft"),
        ("KTO alignment",   "checkpoints/kto"),
        ("KTO-RAFT iter1",  "checkpoints/kto-raft/iter1"),
        ("KTO-RAFT iter2",  "checkpoints/kto-raft/iter2"),
        ("KTO-RAFT iter3",  "checkpoints/kto-raft/iter3"),
    ]

    all_results = {}
    for tag, adapter in stages:
        if not Path(adapter).exists():
            print(f"\n[skip] {tag} — checkpoint not found at {adapter}")
            continue

        print(f"\n{'#'*60}")
        print(f"  Evaluating: {tag}")
        print(f"  Adapter   : {adapter}")
        print(f"{'#'*60}")

        results = run_evaluation(adapter, tag, quick=quick, verbose=verbose)
        all_results[tag] = results

    # Print final comparison
    print(f"\n\n{'#'*70}")
    print(f"  OVERALL PROGRESSION")
    print(f"{'#'*70}")
    print(f"  {'Stage':<25} {'GSM8K':>8} {'MMLU':>8} {'StratQA':>8}")
    print(f"  {'-'*55}")
    for tag, res in all_results.items():
        g = res.get("gsm8k", {}).get("accuracy", 0)
        m = res.get("mmlu",  {}).get("accuracy", 0)
        s = res.get("strategy_qa", {}).get("accuracy", 0)
        print(f"  {tag:<25} {g:>7.1f}% {m:>7.1f}% {s:>7.1f}%")

    output_path = "logs/all_stages_eval.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results saved → {output_path}")


# ─── Single Adapter Evaluation ────────────────────────────────────────────────

def run_evaluation(adapter_path: str, tag: str, quick: bool, verbose: bool) -> dict:
    from mlx_lm import load as mlx_load

    print(f"\nLoading model: {config.BASE_MODEL} + adapter: {adapter_path}")
    model, tokenizer = mlx_load(config.BASE_MODEL, adapter_path=adapter_path or None)

    n_gsm8k    = 100 if quick else config.EVAL_GSM8K_SAMPLES
    n_mmlu     = 10  if quick else config.EVAL_MMLU_SAMPLES
    n_strategy = 50  if quick else config.EVAL_STRATEGY_SAMPLES

    results = {}
    t_total = time.time()

    results["gsm8k"]       = evaluate_gsm8k(model, tokenizer, n_gsm8k, verbose)
    results["mmlu"]        = evaluate_mmlu(model, tokenizer, n_mmlu, verbose)
    results["strategy_qa"] = evaluate_strategyqa(model, tokenizer, n_strategy, verbose)

    results["meta"] = {
        "adapter":      adapter_path,
        "tag":          tag,
        "total_time_s": round(time.time() - t_total, 1),
        "quick_mode":   quick,
    }

    print_comparison_table(tag, results)

    # Save
    out_path = f"logs/eval_{tag.replace(' ', '_').replace('/', '-')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Full results → {out_path}")

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified Benchmark Evaluation Suite")
    parser.add_argument("--adapter",     default="checkpoints/kto-raft/iter3",
                        help="Path to LoRA adapter to evaluate.")
    parser.add_argument("--tag",         default="KTO-RAFT",
                        help="Label for this model in output tables.")
    parser.add_argument("--quick",       action="store_true",
                        help="Run 100/10/50 samples instead of full benchmark (smoke test).")
    parser.add_argument("--all-stages",  action="store_true", dest="all_stages",
                        help="Evaluate all pipeline checkpoints (SFT, KTO, RAFT iters).")
    parser.add_argument("--verbose",     action="store_true",
                        help="Print per-example warnings.")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)

    if args.all_stages:
        evaluate_all_stages(quick=args.quick, verbose=args.verbose)
    else:
        if not Path(args.adapter).exists() and args.adapter != "":
            print(f"❌ Adapter not found: {args.adapter}")
            print("   Run with --all-stages to auto-discover checkpoints.")
            sys.exit(1)
        run_evaluation(
            adapter_path=args.adapter if args.adapter else None,
            tag=args.tag,
            quick=args.quick,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
