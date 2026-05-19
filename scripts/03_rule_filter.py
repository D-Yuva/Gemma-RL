"""
Data Segregator and Curriculum Generator.
Sorts formatted examples by the pre-computed difficulty and splits into train/val.
Input:  data/processed/all_formatted.jsonl
Output: data/splits/train.jsonl, data/splits/valid.jsonl
"""
import json
from pathlib import Path
import sys
import os

# Add parent dir to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def main():
    Path(config.DATA_SPLITS).mkdir(parents=True, exist_ok=True)
    input_file = Path(config.DATA_PROCESSED) / "all_formatted.jsonl"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found. Run 02_format_data.py first.")
        return

    print(f"Loading formatted examples from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        all_examples = [json.loads(l) for l in f]

    print(f"Loaded {len(all_examples)} examples. Segregating by difficulty...")
    
    # ── Sort by difficulty (curriculum learning: easy first)
    # The difficulty was already precisely computed in 02_format_data.py
    all_examples.sort(key=lambda x: x.get("difficulty", 0.5))

    # Hold out last 500 (or 10%) as internal val set
    val_size = min(500, max(1, len(all_examples) // 10))
    train_set = all_examples[:len(all_examples)-val_size]
    val_set   = all_examples[len(all_examples)-val_size:]

    train_out = Path(config.DATA_SPLITS) / "train.jsonl"
    val_out = Path(config.DATA_SPLITS) / "valid.jsonl"

    print(f"Saving splits to {config.DATA_SPLITS}...")
    with open(train_out, "w", encoding="utf-8") as f:
        for ex in train_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_out, "w", encoding="utf-8") as f:
        for ex in val_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n✅ Data Segregation Complete!")
    print(f"   Total Examples: {len(all_examples)}")
    print(f"   Train: {len(train_set)} · Val: {len(val_set)}")
    print(f"   Curriculum sorted from difficulty {all_examples[0]['difficulty']:.3f} to {all_examples[-1]['difficulty']:.3f}")
    print(f"\nNext: Run SFT training with scripts/04_sft_train.sh")

if __name__ == "__main__":
    main()
