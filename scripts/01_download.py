"""Download and verify the three target training datasets."""
import os
import sys
from datasets import load_dataset
from pathlib import Path

# Add parent dir to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

Path(config.DATA_RAW).mkdir(parents=True, exist_ok=True)

datasets_to_load = [
    ("gsm8k", "main"),
    ("meta-math/MetaMathQA", None),
    ("hotpot_qa", "distractor"),
]

for name, subset in datasets_to_load:
    try:
        print(f"Downloading {name}...")
        ds = load_dataset(name, subset, cache_dir=config.DATA_RAW, trust_remote_code=True)
        splits = {k: len(v) for k, v in ds.items()}
        print(f"✅ {name}: {splits}")
    except Exception as e:
        print(f"❌ {name}: {e}")

print(f"\nAll downloads complete. Data cached in {config.DATA_RAW}/")
