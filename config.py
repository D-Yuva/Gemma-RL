# config.py - Configuration for the Qwen Teacher data segregation pipeline

BASE_MODEL     = "google/gemma-4-E4B-it"

# LM Studio uses an OpenAI-compatible endpoint. 
# Ensure LM Studio's Local Server is running on port 1234
LM_STUDIO_URL  = "http://localhost:1234/v1"
QWEN_MODEL     = "qwen/qwen3-14b"  # Adjust if the name differs in your LM Studio model list

DATA_RAW       = "data/raw"
DATA_PROCESSED = "data/processed"
DATA_SPLITS    = "data/splits"

# For formatting, how many samples to use to keep training fast
SAMPLES_METAMATH = 30000
SAMPLES_ORCAMATH = 10000
SAMPLES_HOTPOTQA = 5000

# SFT Training Hyperparameters
SFT_ITERS        = 5000
SFT_LR           = 2e-5
SFT_BATCH_SIZE   = 1
SFT_LORA_RANK    = 32
SFT_MAX_SEQ_LEN  = 512
SFT_VAL_BATCHES  = 25
SFT_SAVE_EVERY   = 500
