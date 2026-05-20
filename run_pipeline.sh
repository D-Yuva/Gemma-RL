#!/bin/bash
# run_pipeline.sh — Full KTO-OMEGA Pipeline Runner
#
# Runs all 8 stages sequentially with checkpoint validation between stages.
# Hardware: Mac M4 Pro (Apple Silicon, 24GB unified memory)
# Estimated total training time: 18–28 hours
#
# Usage:
#   bash run_pipeline.sh                    # Full pipeline
#   bash run_pipeline.sh --skip-sft         # Skip SFT (already trained)
#   bash run_pipeline.sh --start-from kto   # Start from KTO stage
#   bash run_pipeline.sh --dry-run          # Test pipeline without model generation

set -e   # Exit immediately on error

# ─── Parse flags ──────────────────────────────────────────────────────────────
SKIP_DOWNLOAD=false
SKIP_FORMAT=false
SKIP_FILTER=false
SKIP_SFT=false
SKIP_PRM=true          # PRM is optional; skip by default
DRY_RUN=false
START_FROM=""

for arg in "$@"; do
  case $arg in
    --skip-download) SKIP_DOWNLOAD=true ;;
    --skip-sft)      SKIP_SFT=true ;;
    --include-prm)   SKIP_PRM=false ;;
    --dry-run)       DRY_RUN=true ;;
    --start-from=*)  START_FROM="${arg#*=}" ;;
    --help)
      echo "Usage: bash run_pipeline.sh [flags]"
      echo "  --skip-download   Skip dataset download (01)"
      echo "  --skip-sft        Skip SFT training (04) — use existing checkpoint"
      echo "  --include-prm     Include PRM training (05b) — disabled by default"
      echo "  --dry-run         Use synthetic undesirable data; skip real generation"
      echo "  --start-from=X    Skip to stage X: data|sft|kto|raft|eval"
      exit 0 ;;
  esac
done

# ─── Helpers ──────────────────────────────────────────────────────────────────
log() { echo -e "\n\033[1;36m[$(date '+%H:%M:%S')] $1\033[0m"; }
ok()  { echo -e "\033[1;32m  ✅ $1\033[0m"; }
err() { echo -e "\033[1;31m  ❌ $1\033[0m"; exit 1; }

check_checkpoint() {
  local path="$1"
  local name="$2"
  if [ -d "$path" ] && [ "$(ls -A "$path" 2>/dev/null)" ]; then
    ok "$name checkpoint found: $path"
    return 0
  else
    echo "  [warn] Checkpoint not found: $path"
    return 1
  fi
}

mkdir -p logs checkpoints data/raw data/processed data/splits

# ─── Stage 01: Download ───────────────────────────────────────────────────────
if [ "$START_FROM" = "data" ] || [ "$START_FROM" = "" ]; then
  if [ "$SKIP_DOWNLOAD" = false ]; then
    log "Stage 01: Downloading datasets (GSM8K, MetaMathQA, HotpotQA)..."
    python scripts/01_download.py 2>&1 | tee logs/01_download.log
    ok "Downloads complete"
  else
    log "Stage 01: [SKIPPED] Download"
  fi

  # ─── Stage 02: Format ────────────────────────────────────────────────────────
  log "Stage 02: Formatting data into CoT JSONL..."
  python scripts/02_format_data.py 2>&1 | tee logs/02_format.log
  ok "Formatting complete → data/processed/all_formatted.jsonl"

  # ─── Stage 03: Filter ────────────────────────────────────────────────────────
  log "Stage 03: Curriculum sorting and train/valid split..."
  python scripts/03_rule_filter.py 2>&1 | tee logs/03_filter.log
  ok "Splits ready → data/splits/train.jsonl + valid.jsonl"
fi

# ─── Stage 04: SFT ───────────────────────────────────────────────────────────
if [ "$START_FROM" = "sft" ] || [ "$START_FROM" = "" ]; then
  if [ "$SKIP_SFT" = false ]; then
    log "Stage 04: SFT Training (Gemma 4 E4B, MLX LoRA)..."
    echo "[IMPORTANT] Estimated time: 4–8 hours on M4 Pro"
    echo "[IMPORTANT] Close all other apps to maximise available memory"
    bash scripts/04_sft_train.sh
    check_checkpoint "checkpoints/sft" "SFT" || err "SFT training failed — no checkpoint found"
    ok "SFT complete → checkpoints/sft/"
  else
    log "Stage 04: [SKIPPED] SFT — using existing checkpoint"
    check_checkpoint "checkpoints/sft" "SFT" || err "No SFT checkpoint found at checkpoints/sft/. Remove --skip-sft."
  fi
fi

# ─── Stage 05b: PRM (optional) ───────────────────────────────────────────────
if [ "$SKIP_PRM" = false ]; then
  log "Stage 05b: PRM Data Preparation..."
  python scripts/05b_prm_train.py 2>&1 | tee logs/05b_prm_data.log
  ok "PRM data ready. Run the printed training command then continue."
  echo ""
  echo "⏸  PAUSED: PRM training must be run manually (command printed above)."
  echo "    After PRM training completes, re-run this script with --start-from=kto"
  exit 0
fi

# ─── Stage 05: KTO Data Prep ─────────────────────────────────────────────────
if [ "$START_FROM" = "kto" ] || [ "$START_FROM" = "" ]; then
  log "Stage 05: Building KTO binary-label dataset..."
  if [ "$DRY_RUN" = true ]; then
    python scripts/05_prepare_kto_data.py --dry-run 2>&1 | tee logs/05_kto_data.log
  else
    python scripts/05_prepare_kto_data.py --sft-adapter checkpoints/sft 2>&1 | tee logs/05_kto_data.log
  fi

  # Verify output exists
  if [ ! -f "data/splits/kto_train.jsonl" ]; then
    err "KTO dataset not created. Check logs/05_kto_data.log"
  fi
  ok "KTO dataset ready → data/splits/kto_train.jsonl"

  # ─── Stage 06b: KTO Training (MLX native) ──────────────────────────────────
  log "Stage 06b: KTO Alignment Training — MLX Native..."
  echo "[INFO] Estimated time: 6–10 hours on M4 Pro"
  python scripts/06b_kto_train_mlx.py \
    --data            data/splits/kto_train.jsonl \
    --sft-adapter     checkpoints/sft \
    --output          checkpoints/kto \
    2>&1 | tee logs/06b_kto_mlx.log

  check_checkpoint "checkpoints/kto" "KTO" || {
    log "MLX KTO failed. Trying TRL fallback..."
    python scripts/06_kto_train_trl.py \
      --data            data/splits/kto_train.jsonl \
      --sft-adapter     checkpoints/sft \
      --output          checkpoints/kto \
      2>&1 | tee logs/06_kto_trl.log
    check_checkpoint "checkpoints/kto" "KTO (TRL)" || err "KTO training failed on both MLX and TRL paths"
  }
  ok "KTO alignment complete → checkpoints/kto/"

  # Quick evaluation after KTO
  log "Quick eval after KTO alignment (100 samples)..."
  python scripts/08_evaluate.py \
    --adapter checkpoints/kto \
    --tag     "KTO-aligned" \
    --quick   \
    2>&1 | tee logs/eval_kto_quick.log
fi

# ─── Stage 07: KTO-RAFT ──────────────────────────────────────────────────────
if [ "$START_FROM" = "raft" ] || [ "$START_FROM" = "" ]; then
  log "Stage 07: KTO-RAFT Self-Improvement (3 iterations)..."
  echo "[INFO] Estimated time: 12–18 hours on M4 Pro (4h per iteration)"

  PRM_FLAG=""
  if [ -d "checkpoints/prm" ] && [ "$SKIP_PRM" = false ]; then
    PRM_FLAG="--prm-adapter checkpoints/prm"
    log "PRM adapter found — enabling step-level scoring"
  else
    PRM_FLAG="--use-orm-only"
    log "Using rule-based ORM (no PRM)"
  fi

  python scripts/07_kto_raft.py \
    --kto-adapter checkpoints/kto \
    $PRM_FLAG \
    --n-problems  500 \
    --eval-samples 200 \
    2>&1 | tee logs/07_raft.log

  ok "KTO-RAFT complete → checkpoints/kto-raft/"
fi

# ─── Stage 08: Final Evaluation ──────────────────────────────────────────────
if [ "$START_FROM" = "eval" ] || [ "$START_FROM" = "" ]; then
  log "Stage 08: Final Benchmark Evaluation..."

  # Find best RAFT checkpoint
  BEST_CKPT="checkpoints/kto-raft/iter3"
  if [ ! -d "$BEST_CKPT" ]; then
    BEST_CKPT="checkpoints/kto"
    echo "[info] iter3 not found, evaluating KTO checkpoint"
  fi

  python scripts/08_evaluate.py \
    --adapter    "$BEST_CKPT" \
    --tag        "KTO-RAFT-final" \
    2>&1 | tee logs/08_eval_final.log

  # Full stage comparison
  python scripts/08_evaluate.py \
    --all-stages \
    2>&1 | tee logs/08_eval_all_stages.log

  ok "Evaluation complete → logs/eval_*.json"
fi

# ─── Done ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           KTO-OMEGA PIPELINE COMPLETE ✅                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Checkpoints : checkpoints/kto-raft/iter3/                   ║"
echo "║  Eval results: logs/eval_KTO-RAFT-final.json                 ║"
echo "║  RAFT log    : logs/raft_log.json                             ║"
echo "║  Stage comp. : logs/all_stages_eval.json                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
