# config.py - Central configuration for the full OMEGA+KTO pipeline
# Covers: Data paths, SFT, PRM, KTO alignment, KTO-RAFT, and Evaluation

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

# ── PRM (Process Reward Model) Hyperparameters ────────────────────────────────
# Trained on PRM800K step-level labels
PRM_ITERS        = 2000
PRM_LR           = 1e-5
PRM_BATCH_SIZE   = 4
PRM_LORA_RANK    = 16
PRM_MAX_SEQ_LEN  = 512
PRM_SAVE_EVERY   = 250

# ── KTO Alignment Hyperparameters ─────────────────────────────────────────────
# Replaces DPO. β=0.1 for SFT-pretrained model (lower = less conservative).
# lr=5e-6 is ~10x higher than DPO's typical 5e-7 (KTO is less lr-sensitive).
KTO_BETA         = 0.1           # Prospect-theory risk-aversion parameter
KTO_LR           = 5e-6          # Learning rate (AdamW)
KTO_BATCH_SIZE   = 2             # Minimum 2 required for KL estimate
KTO_GRAD_ACCUM   = 8             # Effective batch = 2 × 8 = 16
KTO_ITERS        = 3000
KTO_LORA_RANK    = 32
KTO_NUM_LORA_LAYERS = 16
KTO_MAX_SEQ_LEN  = 512
KTO_SAVE_EVERY   = 300
KTO_LOG_EVERY    = 20
# λD/λU: set to 0 to auto-compute from dataset ratio (recommended)
# Formula: λD = 1.5 × (n_undesirable / n_desirable), λU = 1.0
LAMBDA_D         = 0             # 0 = auto-compute
LAMBDA_U         = 0             # 0 = auto-compute

# ── KTO-RAFT Self-Improvement Hyperparameters ─────────────────────────────────
# RAFT: generate K solutions → ORM filter → KTO update → repeat
RAFT_K               = 4         # Solutions sampled per problem
RAFT_N_ITERS         = 3         # Number of RAFT outer iterations
RAFT_TEMP            = 0.8       # Sampling temperature for generation
RAFT_N_PROBLEMS      = 500       # Problems used per RAFT iteration
RAFT_KTO_ITERS       = 1000      # KTO update steps per RAFT iteration
RAFT_KTO_LR          = 5e-6
RAFT_MAX_GEN_TOKENS  = 512

# ── Evaluation Hyperparameters ────────────────────────────────────────────────
EVAL_GSM8K_SAMPLES   = 1319      # Full GSM8K test set
EVAL_MMLU_SAMPLES    = 100       # Subset per MMLU subject (57 subjects)
EVAL_STRATEGY_SAMPLES = 500      # StrategyQA samples
EVAL_TEMP            = 0.0       # Greedy decoding for evaluation
EVAL_MAX_TOKENS      = 512
