"""
Format the 3 requested datasets into a unified CoT JSONL format.
Output: data/processed/all_formatted.jsonl
"""
import json
import random
import re
from datasets import load_dataset
from pathlib import Path
import sys
import os

# Add parent dir to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

random.seed(42)
Path(config.DATA_PROCESSED).mkdir(parents=True, exist_ok=True)

# ── The universal template every formatter must produce ───────
# Every example MUST have: prompt, completion, answer, task, source, difficulty
# completion MUST contain <think>...</think><answer>...</answer>

def score_gsm8k(question, steps):
    # 1. Linguistic & Entity Complexity
    q_words = len(question.split())
    # Count unique entities (proper nouns/capitalized words in middle of sentence)
    entities = len(re.findall(r'\b[A-Z][a-z]+\b', question))
    
    # 2. Logic Complexity (comparative phrases)
    comparatives = len(re.findall(r'\b(half|twice|thrice|more than|less than|percentage|ratio|each|every|remaining)\b', question, re.I))
    
    # 3. Numerical Density
    digits = sum(c.isdigit() for c in question)
    digit_density = digits / len(question) if len(question) > 0 else 0
    # Penalty for large numbers (indicates more complex arithmetic)
    large_nums = len(re.findall(r'\b\d{3,}\b', question))
    
    # 4. Reasoning Step Depth
    calc_steps = len(re.findall(r'<<.*?>>', steps))
    ops = len(re.findall(r'[+\-*/=]', steps))
    
    # 5. Length of solution
    s_words = len(steps.split())

    # Weighted Score (0.1 - 1.0)
    score = (
        min(q_words / 60, 1.0) * 0.10 +      # Question length
        min(entities / 5, 1.0) * 0.05 +      # Entity density
        min(comparatives / 4, 1.0) * 0.15 +  # Logic complexity
        min(digit_density * 10, 1.0) * 0.05 + # Digit density
        min(large_nums / 3, 1.0) * 0.05 +    # Large numbers
        min(calc_steps / 8, 1.0) * 0.35 +    # ACTUAL reasoning steps (Primary)
        min(ops / 20, 1.0) * 0.15 +          # Operator count
        min(s_words / 200, 1.0) * 0.10       # Solution verbosity
    )
    return round(score, 3)

def score_metamath(query, response):
    # 1. Structural Complexity
    q_words = len(query.split())
    r_words = len(response.split())
    
    # 2. Variable Interaction
    # Count unique variables (x, y, z, a, b, c, n, k)
    vars = len(set(re.findall(r'\b([a-z])\b', query)))
    
    # 3. Advanced Math Detection (Trig, Calculus, etc)
    advanced_keywords = len(re.findall(r'\\(sin|cos|tan|log|ln|sqrt|sum|int|lim|theta|pi|phi|alpha|beta|gamma)\b', response))
    
    # 4. LaTeX Complexity
    latex_chars = len(re.findall(r'[\{\}\\\^_\$]', response))
    latex_density = latex_chars / len(response) if len(response) > 0 else 0
    # Nested LaTeX structures (e.g., fractions within fractions)
    nesting = len(re.findall(r'\\frac\{', response))
    
    # 5. Step indicators (Logical flow)
    indicators = len(re.findall(r'(Step|Then|Therefore|So|Thus|Hence|Since|Assume|Let|Case)', response, re.I))
    
    # Weighted Score
    score = (
        min(q_words / 120, 1.0) * 0.05 +
        min(vars / 4, 1.0) * 0.15 +
        min(advanced_keywords / 3, 1.0) * 0.20 + # Bonus for high-level math
        min(latex_density * 15, 1.0) * 0.15 +
        min(nesting / 2, 1.0) * 0.10 +           # Structural depth
        min(indicators / 10, 1.0) * 0.20 +       # Reasoning chain
        min(r_words / 400, 1.0) * 0.15           # Length
    )
    return round(score, 3)

def score_hotpot(ex):
    # 1. Native Metadata
    level_map = {"easy": 0.2, "medium": 0.5, "hard": 0.8}
    base_level = level_map.get(ex.get("level"), 0.5)
    
    # 2. Multi-Hop Diversity
    supporting_facts = ex.get("supporting_facts", {})
    titles = set(supporting_facts.get("title", []))
    doc_diversity = len(titles) # How many distinct documents are involved
    
    # 3. Question Complexity
    question = ex.get("question", "")
    q_words = len(question.split())
    # Complex multi-entity questions (Capitalized words)
    entities = len(re.findall(r'\b[A-Z][a-z]+\b', question))
    
    # 4. Search Complexity
    type_bonus = 0.2 if ex.get("type") == "bridge" else 0.0
    
    total_context_words = 0
    context = ex.get("context", {})
    for sentences in context.get("sentences", []):
        for s in sentences:
            total_context_words += len(s.split())
            
    num_facts = len(supporting_facts.get("title", []))
    
    # Weighted Score
    score = (
        base_level * 0.4 +
        min(doc_diversity / 3, 1.0) * 0.2 + # Multi-source requirement
        min(num_facts / 6, 1.0) * 0.15 +    # Reasoning chain density
        min(entities / 4, 1.0) * 0.05 +      # Entity recognition difficulty
        type_bonus +
        min(total_context_words / 2500, 1.0) * 0.1 +
        min(q_words / 30, 1.0) * 0.1
    )
    return round(min(score, 1.0), 3)

def fmt_gsm8k(ex):
    parts   = ex["answer"].split("####")
    steps   = parts[0].strip()
    answer  = parts[1].strip() if len(parts) > 1 else ""
    return {
        "prompt": f"Solve step-by-step:\n{ex['question']}",
        "completion": f"<think>\n{steps}\n</think>\n<answer>{answer}</answer>",
        "answer": answer, "task": "math",
        "source": "gsm8k", "difficulty": score_gsm8k(ex["question"], steps),
    }

def fmt_metamath(ex):
    query    = ex.get("query", "").strip()
    response = ex.get("response", "").strip()
    ans_m = re.search(r'(?:The answer is|\\boxed{|####\s*)([^}.\n]+)', response)
    answer = ans_m.group(1).strip() if ans_m else ""
    return {
        "prompt": f"Solve step-by-step:\n{query}",
        "completion": f"<think>\n{response}\n</think>\n<answer>{answer}</answer>",
        "answer": answer, "task": "math",
        "source": "metamath", "difficulty": score_metamath(query, response),
    }

def fmt_hotpotqa(ex):
    question = ex.get("question", "")
    answer   = ex.get("answer", "")
    context  = ex.get("context", {})
    titles   = context.get("title", [])
    reasoning = f"This requires checking multiple sources.\nSources: {', '.join(titles[:3])}.\nBased on these facts: {answer}"
    
    return {
        "prompt": f"Answer this step-by-step using evidence:\n{question}",
        "completion": f"<think>\n{reasoning}\n</think>\n<answer>{answer}</answer>",
        "answer": answer.lower(), "task": "qa",
        "source": "hotpotqa", "difficulty": score_hotpot(ex)
    }

def main():
    all_examples = []

    # 1. GSM8K
    print("Formatting GSM8K...")
    gsm = load_dataset("gsm8k", "main", cache_dir=config.DATA_RAW)
    gsm_formatted = [fmt_gsm8k(ex) for ex in gsm["train"]]
    all_examples.extend(gsm_formatted)
    print(f"  → {len(gsm_formatted)} GSM8K examples")

    # 2. MetaMathQA
    print("Formatting MetaMathQA...")
    meta = load_dataset("meta-math/MetaMathQA", cache_dir=config.DATA_RAW)
    meta_formatted = []
    subtypes = {}
    for ex in meta["train"]:
        qtype = ex.get("type", "unknown")
        if qtype not in subtypes: subtypes[qtype] = []
        subtypes[qtype].append(ex)

    num_types = len(subtypes)
    samples_per_type = config.SAMPLES_METAMATH // num_types if num_types > 0 else 0
    for qtype, raw_examples in subtypes.items():
        sampled_raw = random.sample(raw_examples, min(samples_per_type, len(raw_examples)))
        formatted = [fmt_metamath(ex) for ex in sampled_raw]
        meta_formatted.extend(formatted)
        print(f"  {qtype}: {len(formatted)} samples")

    all_examples.extend(meta_formatted)
    print(f"  → {len(meta_formatted)} MetaMathQA examples")

    # 3. HotpotQA
    print("Formatting HotpotQA...")
    hotpot = load_dataset("hotpot_qa", "distractor", cache_dir=config.DATA_RAW)
    hotpot_formatted = [fmt_hotpotqa(ex) for ex in hotpot["train"]]
    hotpot_sampled = random.sample(hotpot_formatted, min(config.SAMPLES_HOTPOTQA, len(hotpot_formatted)))
    all_examples.extend(hotpot_sampled)
    print(f"  → {len(hotpot_sampled)} HotpotQA examples")

    # Save
    out_path = Path(config.DATA_PROCESSED) / "all_formatted.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n✅ Total formatted: {len(all_examples)} examples")
    print(f"Saved to {out_path}")
    print("Next: Run 03_qwen_filter.py to sort and split the data.")

if __name__ == "__main__":
    main()
