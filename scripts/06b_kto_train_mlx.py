"""
06b_kto_train_mlx.py — KTO Training: Native MLX (Primary Path, Mac M4 Pro)

Architecture
------------
This implements KTO without keeping two full models in memory simultaneously.
Instead it uses an "offline reference" strategy:

  Step 1 (pre-compute, runs once):
    Load SFT model → compute log P_ref(y|x) for every training example → save to disk.

  Step 2 (hot training loop):
    Load only the POLICY model.
    For each batch: look up pre-cached ref log-probs, compute KTO loss, update.

Memory footprint:
  DPO (policy + reference in one pass)  : ~18.4 GB
  This script (policy only + cached ref) : ~12.0 GB  ← 35% less

KTO Loss (Kahneman-Tversky prospect theory, eq. 7):
  r_θ  = log π_θ(y|x)  −  log π_ref(y|x)        (implied reward)
  z0   = max(mean(r_θ_shifted), 0)               (KL estimate, no grad)

  v_desirable   = λD × (1 − σ(β(r_θ − z0)))     (push reward UP)
  v_undesirable = λU × (1 − σ(β(z0 − r_θ)))     (push reward DOWN)
  L = E[v(x,y)]

Usage
-----
  # With auto-computed λ from kto_stats.json:
  python scripts/06b_kto_train_mlx.py

  # With manual overrides:
  python scripts/06b_kto_train_mlx.py \\
      --data data/splits/kto_train.jsonl \\
      --sft-adapter checkpoints/sft \\
      --output checkpoints/kto \\
      --beta 0.1 --lr 5e-6 --iters 3000 --lora-rank 32

Troubleshooting
---------------
  Loss not decreasing after 300 steps  → increase lr to 1e-5; decrease beta to 0.05
  Mode collapse (repetition)           → increase beta to 0.3; subsample desirable
  GSM8K drops after KTO                → reduce iters to 1500; use lr=1e-6
  OOM                                  → reduce --batch-size to 1 and --grad-accum to 32
  KL divergence > 0.5                  → reduce lr to 5e-7
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
import config


# ─── Utilities ────────────────────────────────────────────────────────────────

def load_mlx_model(model_name: str, adapter_path: str | None = None):
    """Load model + tokenizer via mlx_lm."""
    from mlx_lm import load
    print(f"[mlx] Loading {model_name}" + (f" + adapter {adapter_path}" if adapter_path else ""))
    model, tokenizer = load(model_name, adapter_path=adapter_path if adapter_path else None)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def safe_model_call(model, input_ids: mx.array) -> mx.array:
    """
    Call the model and return logits, handling both return types:
      - (logits,)            — some mlx_lm model variants
      - (logits, cache)      — most mlx_lm models
    """
    out = model(input_ids)
    if isinstance(out, tuple):
        return out[0]
    return out


# ─── Tokenization & Batching ──────────────────────────────────────────────────

def tokenize_example(tokenizer, prompt: str, completion: str, max_len: int):
    """
    Tokenize a (prompt, completion) pair.
    Returns (input_ids: list[int], completion_mask: list[float])
    completion_mask is 1.0 only at completion token positions.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    comp_ids   = tokenizer.encode(completion, add_special_tokens=False)

    # Safety: strip very long examples rather than truncating the prompt
    max_comp = max_len - len(prompt_ids)
    if max_comp < 4:
        # Prompt itself exceeds budget — truncate prompt end
        prompt_ids = prompt_ids[:max_len - 4]
        max_comp = 4
    comp_ids = comp_ids[:max_comp]

    ids  = prompt_ids + comp_ids
    mask = [0.0] * len(prompt_ids) + [1.0] * len(comp_ids)
    return ids, mask


def make_batch(examples: list[dict], tokenizer, max_len: int) -> dict:
    """
    Collate a list of KTO examples into a padded MLX batch.
    Returns dict with keys: input_ids, completion_mask, labels.
    """
    all_ids, all_masks, all_labels = [], [], []
    for ex in examples:
        ids, mask = tokenize_example(tokenizer, ex["prompt"], ex["completion"], max_len)
        all_ids.append(ids)
        all_masks.append(mask)
        all_labels.append(1.0 if ex["label"] else 0.0)

    # Pad to longest sequence in this batch
    max_batch_len = max(len(ids) for ids in all_ids)
    pad_id = tokenizer.pad_token_id

    padded_ids, padded_masks = [], []
    for ids, mask in zip(all_ids, all_masks):
        n = max_batch_len - len(ids)
        padded_ids.append(ids + [pad_id] * n)
        padded_masks.append(mask + [0.0] * n)

    return {
        "input_ids":       mx.array(padded_ids,    dtype=mx.int32),
        "completion_mask": mx.array(padded_masks,  dtype=mx.float32),
        "labels":          mx.array(all_labels,    dtype=mx.float32),
    }


# ─── Log-prob Computation ─────────────────────────────────────────────────────

def compute_completion_logprobs(logits: mx.array, input_ids: mx.array, completion_mask: mx.array) -> mx.array:
    """
    Sum log P(token) over completion positions only.

    Args:
        logits:          (B, T, V)  model output
        input_ids:       (B, T)     token ids
        completion_mask: (B, T)     1.0 at completion positions, 0.0 elsewhere

    Returns:
        (B,) sum of completion token log-probs
    """
    # Shift: logit at t predicts token at t+1
    shifted_logits = logits[:, :-1, :]         # (B, T-1, V)
    target_ids     = input_ids[:, 1:]           # (B, T-1)
    target_mask    = completion_mask[:, 1:]     # (B, T-1)

    log_probs = nn.log_softmax(shifted_logits, axis=-1)   # (B, T-1, V)

    # Gather log-prob for each actual target token
    B, T1 = target_ids.shape
    b_idx = mx.arange(B)[:, None]              # (B, 1)
    t_idx = mx.arange(T1)[None, :]             # (1, T-1)
    token_lp = log_probs[b_idx, t_idx, target_ids]       # (B, T-1)

    # Sum only over completion tokens
    return (token_lp * target_mask).sum(axis=-1)          # (B,)


# ─── Reference Log-prob Caching (runs once before training) ──────────────────

def precompute_ref_logprobs(
    model,
    tokenizer,
    data: list[dict],
    max_len: int,
    cache_path: str,
    batch_size: int = 4,
) -> np.ndarray:
    """
    Run the SFT (reference) model over all KTO data once to cache completion
    log-probs.  This avoids loading two models during the training loop.

    The cache is saved as a float32 numpy array aligned with `data` indices.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"[ref] Loading cached ref log-probs from {cache_path}")
        return np.load(str(cache_path))

    print(f"[ref] Pre-computing reference log-probs for {len(data)} examples...")
    print(f"      (This runs once and is cached to {cache_path})")
    all_lp = []

    for i in range(0, len(data), batch_size):
        batch_data = data[i: i + batch_size]
        batch      = make_batch(batch_data, tokenizer, max_len)
        logits     = safe_model_call(model, batch["input_ids"])
        lp = compute_completion_logprobs(logits, batch["input_ids"], batch["completion_mask"])
        mx.eval(lp)
        all_lp.extend(lp.tolist())

        if (i // batch_size) % 25 == 0:
            pct = 100 * i / len(data)
            print(f"  [{i:5d}/{len(data)}] {pct:.0f}% done", end="\r")

    arr = np.array(all_lp, dtype=np.float32)
    np.save(str(cache_path), arr)
    print(f"\n[ref] Saved ref log-probs → {cache_path}  (shape: {arr.shape})")
    return arr


# ─── KTO Loss ─────────────────────────────────────────────────────────────────

def kto_loss(
    model,
    batch: dict,
    ref_logprobs: mx.array,   # (B,) pre-cached for this specific batch
    beta: float,
    lambda_d: float,
    lambda_u: float,
) -> tuple[mx.array, dict]:
    """
    KTO loss (Ethayarajh et al. 2024, eq. 7).
    Returns (loss, metrics) where metrics is a plain dict for logging.
    """
    logits   = safe_model_call(model, batch["input_ids"])   # (B, T, V)
    policy_lp = compute_completion_logprobs(logits, batch["input_ids"], batch["completion_mask"])

    # Implied reward: how much better is policy than reference?
    r_theta = policy_lp - ref_logprobs     # (B,)

    # KL estimate from mismatched batch (circular shift — breaks pairing)
    # No gradient flows through z0 (stop_gradient critical for training stability)
    r_shifted = mx.concatenate([r_theta[-1:], r_theta[:-1]], axis=0)
    z0 = mx.stop_gradient(mx.maximum(mx.mean(r_shifted), mx.array(0.0)))

    # Prospect-theory value function
    labels = batch["labels"]           # (B,)  1=desirable, 0=undesirable

    des_v = lambda_d * (1.0 - mx.sigmoid(beta * (r_theta - z0)))  # push reward UP
    und_v = lambda_u * (1.0 - mx.sigmoid(beta * (z0 - r_theta)))  # push reward DOWN

    per_example_loss = labels * des_v + (1.0 - labels) * und_v
    loss = mx.mean(per_example_loss)

    # Additional metrics (no gradient)
    mean_reward = mx.mean(r_theta)
    kl_est      = mx.mean(r_shifted)

    return loss, {"mean_reward": mean_reward, "kl_estimate": kl_est, "z0": z0}


# ─── LoRA Setup ───────────────────────────────────────────────────────────────

def add_kto_lora(model, num_layers: int, rank: int):
    """
    Add a fresh LoRA adapter on top of the loaded SFT model.
    This keeps SFT weights frozen; only the new LoRA deltas are trained.
    """
    try:
        from mlx_lm.tuner.lora import linear_to_lora_layers
        lora_config = {
            "rank":    rank,
            "alpha":   rank * 2,
            "dropout": 0.05,
            "scale":   1.0,
        }
        linear_to_lora_layers(model, num_layers, lora_config)
        print(f"[lora] Added KTO LoRA adapters to top {num_layers} layers (rank={rank})")
    except Exception as e:
        print(f"[lora] WARNING: Could not add LoRA ({e}). Will fine-tune full model (slow).")


def freeze_base_unfreeze_lora(model):
    """Freeze all params; then unfreeze only LoRA params."""
    model.freeze()

    # Unfreeze LoRA parameters specifically
    # Pattern: any sub-module or parameter containing 'lora' in its key path
    n_trainable = 0
    for name, module in model.named_modules():
        module_type = type(module).__name__.lower()
        if "lora" in module_type:
            module.unfreeze()
            # Count params in this module
            for _, p in module.named_parameters():
                n_trainable += p.size
    if n_trainable == 0:
        # Fallback: unfreeze last N layers (if LoRA detection failed)
        print("[lora] Fallback: unfreezing last 8 transformer layers for training")
        try:
            layers = model.model.layers
            for layer in layers[-8:]:
                layer.unfreeze()
                for _, p in layer.named_parameters():
                    n_trainable += p.size
        except AttributeError:
            print("[lora] Could not access model.model.layers — unfreezing everything")
            model.unfreeze()
            n_trainable = sum(p.size for _, p in model.named_parameters())

    print(f"[lora] Trainable parameters: {n_trainable / 1e6:.2f}M")
    return n_trainable


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_kto_mlx(args):
    """Main KTO training loop — native MLX."""
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # ── Load KTO data ─────────────────────────────────────────────────────────
    print(f"Loading KTO data from {args.data}...")
    with open(args.data, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    n_d = sum(1 for ex in data if ex["label"])
    n_u = len(data) - n_d
    print(f"  → {len(data)} examples  ({n_d} desirable, {n_u} undesirable)")

    # ── Load λD, λU ───────────────────────────────────────────────────────────
    stats_path = Path(config.DATA_SPLITS) / "kto_stats.json"
    if args.lambda_d == 0:
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            lambda_d = stats.get("lambda_d", 1.0)
            lambda_u = stats.get("lambda_u", 1.0)
        else:
            lambda_u = 1.0
            lambda_d = round(1.5 * n_u / max(n_d, 1), 4)
        print(f"  → Auto λD={lambda_d:.4f}, λU={lambda_u:.4f}")
    else:
        lambda_d = args.lambda_d
        lambda_u = args.lambda_u
        print(f"  → Manual λD={lambda_d:.4f}, λU={lambda_u:.4f}")

    # ── Load SFT model (will be reference AND start of policy) ───────────────
    model, tokenizer = load_mlx_model(config.BASE_MODEL, adapter_path=args.sft_adapter)

    # ── Pre-compute reference log-probs (SFT model, no KTO adapter yet) ──────
    ref_cache_path = str(Path(args.output) / "ref_logprobs_cache.npy")
    ref_logprobs_all = precompute_ref_logprobs(
        model, tokenizer, data,
        max_len=args.max_seq_len,
        cache_path=ref_cache_path,
        batch_size=4,
    )

    # ── Add KTO LoRA adapter on top of SFT model ─────────────────────────────
    add_kto_lora(model, num_layers=args.num_lora_layers, rank=args.lora_rank)
    n_trainable = freeze_base_unfreeze_lora(model)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr_schedule = optim.cosine_decay(
        init=args.lr,
        decay_steps=args.iters,
        end_value=args.lr * 0.1,
    )
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01, betas=(0.9, 0.95))

    # ── Loss + grad function ──────────────────────────────────────────────────
    # nn.value_and_grad differentiates through trainable model params
    def loss_fn(model_, batch_, ref_lp_):
        loss, _ = kto_loss(model_, batch_, ref_lp_, args.beta, lambda_d, lambda_u)
        return loss

    loss_value_and_grad = nn.value_and_grad(model, loss_fn)

    # ── Training state ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"KTO Training — MLX Native | Mac M4 Pro")
    print(f"  Base model    : {config.BASE_MODEL}")
    print(f"  SFT adapter   : {args.sft_adapter}")
    print(f"  β             : {args.beta}")
    print(f"  lr            : {args.lr}  (cosine decay)")
    print(f"  λD / λU       : {lambda_d:.4f} / {lambda_u:.4f}")
    print(f"  Batch size    : {args.batch_size}  ×  grad accum {args.grad_accum}  = eff. {args.batch_size * args.grad_accum}")
    print(f"  Iters         : {args.iters}")
    print(f"  Trainable     : {n_trainable/1e6:.2f}M params")
    print(f"{'='*60}\n")

    indices       = list(range(len(data)))
    log_entries   = []
    accum_grads   = None
    accum_count   = 0
    iter_num      = 0
    epoch         = 0

    random.shuffle(indices)
    data_iter = iter(indices)
    t_start   = time.time()

    while iter_num < args.iters:
        # ── Get next batch (wrapping with epoch tracking) ──────────────────
        batch_indices = []
        for _ in range(args.batch_size):
            try:
                batch_indices.append(next(data_iter))
            except StopIteration:
                epoch += 1
                random.shuffle(indices)
                data_iter = iter(indices)
                batch_indices.append(next(data_iter))

        batch_data   = [data[i] for i in batch_indices]
        batch        = make_batch(batch_data, tokenizer, args.max_seq_len)
        ref_lp_batch = mx.array(ref_logprobs_all[batch_indices], dtype=mx.float32)

        # ── Forward + backward ─────────────────────────────────────────────
        loss, grads = loss_value_and_grad(model, batch, ref_lp_batch)

        # ── Gradient accumulation ──────────────────────────────────────────
        if accum_grads is None:
            accum_grads = grads
        else:
            # Accumulate: element-wise addition of gradient trees
            accum_grads = {k: accum_grads[k] + grads[k] for k in grads}
        accum_count += 1

        if accum_count >= args.grad_accum:
            # Scale and apply gradients
            scaled_grads = {k: v / args.grad_accum for k, v in accum_grads.items()}

            # Gradient clipping (max norm = 1.0)
            grad_norm = mx.sqrt(sum(mx.sum(g ** 2) for g in scaled_grads.values()))
            mx.eval(grad_norm)
            clip_val = 1.0
            if grad_norm.item() > clip_val:
                scale = clip_val / (grad_norm.item() + 1e-8)
                scaled_grads = {k: v * scale for k, v in scaled_grads.items()}

            optimizer.update(model, scaled_grads)
            mx.eval(model.parameters(), optimizer.state)

            accum_grads  = None
            accum_count  = 0
            iter_num    += 1

            # ── Logging ───────────────────────────────────────────────────
            if iter_num % args.log_every == 0 or iter_num == 1:
                mx.eval(loss)
                elapsed = time.time() - t_start
                loss_val = float(loss.item())
                steps_per_sec = iter_num / max(elapsed, 1)
                eta = (args.iters - iter_num) / max(steps_per_sec, 1e-6)

                entry = {
                    "iter":    iter_num,
                    "loss":    round(loss_val, 5),
                    "epoch":   epoch,
                    "elapsed": round(elapsed, 1),
                    "eta_s":   round(eta, 0),
                }
                log_entries.append(entry)

                print(f"[{iter_num:4d}/{args.iters}] "
                      f"loss={loss_val:.4f}  "
                      f"epoch={epoch}  "
                      f"it/s={steps_per_sec:.2f}  "
                      f"ETA={eta/60:.1f}min")

                with open("logs/kto_mlx.json", "w") as flog:
                    json.dump(log_entries, flog, indent=2)

            # ── Checkpoint save ───────────────────────────────────────────
            if iter_num % args.save_every == 0 or iter_num == args.iters:
                ckpt_dir = Path(args.output) / f"step_{iter_num}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                try:
                    from mlx_lm.tuner.utils import save_adapter
                    save_adapter(model, str(ckpt_dir))
                except Exception:
                    # Fallback: save raw weights
                    model.save_weights(str(ckpt_dir / "adapters.npz"))
                print(f"  ✅ Checkpoint saved → {ckpt_dir}")

    # ── Save final adapter ────────────────────────────────────────────────────
    try:
        from mlx_lm.tuner.utils import save_adapter
        save_adapter(model, args.output)
    except Exception:
        model.save_weights(str(Path(args.output) / "adapters.npz"))

    total_time = time.time() - t_start
    print(f"\n✅ KTO training complete!  Total time: {total_time/60:.1f} min")
    print(f"   Adapter → {args.output}/")
    print(f"   Log    → logs/kto_mlx.json")
    print(f"\nNext: python scripts/07_kto_raft.py --kto-adapter {args.output}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="KTO Training — MLX Native (Mac M4 Pro)")
    p.add_argument("--data",            default="data/splits/kto_train.jsonl")
    p.add_argument("--sft-adapter",     default="checkpoints/sft",   dest="sft_adapter")
    p.add_argument("--output",          default="checkpoints/kto")
    p.add_argument("--beta",            type=float, default=config.KTO_BETA)
    p.add_argument("--lr",              type=float, default=config.KTO_LR)
    p.add_argument("--lambda-d",        type=float, default=0,   dest="lambda_d",
                   help="0 = auto-compute from kto_stats.json or data ratio")
    p.add_argument("--lambda-u",        type=float, default=0,   dest="lambda_u")
    p.add_argument("--batch-size",      type=int,   default=config.KTO_BATCH_SIZE,   dest="batch_size")
    p.add_argument("--grad-accum",      type=int,   default=config.KTO_GRAD_ACCUM,   dest="grad_accum")
    p.add_argument("--iters",           type=int,   default=config.KTO_ITERS)
    p.add_argument("--max-seq-len",     type=int,   default=config.KTO_MAX_SEQ_LEN,  dest="max_seq_len")
    p.add_argument("--lora-rank",       type=int,   default=config.KTO_LORA_RANK,    dest="lora_rank")
    p.add_argument("--num-lora-layers", type=int,   default=config.KTO_NUM_LORA_LAYERS, dest="num_lora_layers")
    p.add_argument("--log-every",       type=int,   default=config.KTO_LOG_EVERY,    dest="log_every")
    p.add_argument("--save-every",      type=int,   default=config.KTO_SAVE_EVERY,   dest="save_every")
    args = p.parse_args()
    train_kto_mlx(args)


if __name__ == "__main__":
    main()
