"""
================================================================================
SAE Training for LLaVA-NeXT Llama3-8B Vision Tower (v2)
================================================================================

Trains 3 TopK Sparse Autoencoders on CLIP ViT-L/14-336 hidden states at
layers 12, 17, and 22 (0-indexed) of the vision encoder inside
llava-hf/llama3-llava-next-8b-hf.

Improvements over v1 (vision-sae-train.py):
  1.  Correct model  — Extracts from the LLaVA-NeXT vision tower, NOT LLaVA-1.5
  2.  Multi-layer    — Trains separate SAEs for layers 12, 17, 22
  3.  Better init    — W_enc initialised as transpose of W_dec (prevents dead
                       latents, per OpenAI scaling-SAE paper)
  4.  Preprocessing  — Subtracts dataset mean and divides by mean L2 norm
                       before training, making MSE comparable across layers
  5.  Dead features  — Tracks features with zero activations in a sliding
                       window; reports dead-feature count every log step
  6.  L0 tracking    — Reports mean number of active features per token
  7.  Explained var  — Reports 1 - Var(x-x̂)/Var(x) alongside MSE

Architecture per SAE:
  TopK SAE  (d_model=1024 → d_sae=32768, k=32)
  Encoder:  h = TopK( ReLU( W_enc · (x_norm) + b_enc ), k )
  Decoder:  x̂_norm = W_dec · h + b_dec
  Loss:     MSE = mean( ||x_norm - x̂_norm||² )

Usage:
  # Phase 1 — extract activations (GPU, ~2–3 hrs for 50k images)
  python train_vision_sae_v2.py --phase extract

  # Phase 2 — compute preprocessing stats (CPU OK, ~5 min)
  python train_vision_sae_v2.py --phase stats

  # Phase 3 — train SAEs (GPU, ~1–2 hrs per layer)
  python train_vision_sae_v2.py --phase train

  # All three phases sequentially
  python train_vision_sae_v2.py --phase all
================================================================================
"""

import os
import math
import glob
import json
import argparse
import gc
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
class Config:
    # ── Model ──────────────────────────────────────────────────────────────────
    LLAVA_MODEL_ID  = "llava-hf/llama3-llava-next-8b-hf"
    # The vision tower inside this model is CLIP ViT-L/14-336px.
    # LLaVA-NeXT freezes the vision encoder during training, so the weights
    # are identical to the standalone checkpoint below.  We load the
    # standalone version to avoid pulling 16 GB of LLM weights.
    VISION_MODEL_ID = "openai/clip-vit-large-patch14-336"
    # Set True to extract vision tower from the full LLaVA model instead.
    # Produces identical activations but requires ~16 GB CPU RAM.
    LOAD_FROM_LLAVA = False

    D_MODEL = 1024                       # CLIP ViT-L hidden dim

    # ── Layers to train SAEs on ────────────────────────────────────────────────
    # 0-indexed encoder layer indices.
    # hidden_states[i+1] = output of encoder.layers[i]
    # Layer 22 = hidden_states[-2] = penultimate layer (what LLaVA reads).
    TARGET_LAYERS = [12, 17, 22]

    # ── SAE architecture ───────────────────────────────────────────────────────
    EXPANSION_FACTOR = 32
    D_SAE            = D_MODEL * EXPANSION_FACTOR   # 32 768
    TOP_K            = 32

    # ── Activation extraction ──────────────────────────────────────────────────
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE        = torch.float16 if torch.cuda.is_available() else torch.float32
    MAX_IMAGES   = 50_000
    MAX_ZEBRAS   = 1_000                 # oversample zebra class
    ZEBRA_LABEL  = 340                   # ImageNet class id for zebra
    CHUNK_SIZE   = 500_000               # activation vectors per chunk file
    ACT_DIR      = "./activations_v2"

    # ── SAE training ───────────────────────────────────────────────────────────
    BATCH_SIZE    = 4096
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY  = 0.0
    EPOCHS        = 1
    CKPT_DIR      = "./sae_checkpoints_v2"
    LOG_EVERY     = 50                   # log metrics every N batches
    DEAD_WINDOW   = 50_000               # feature must activate within this
                                         # many tokens to be considered alive
    SEED = 42


# Create directories
os.makedirs(Config.ACT_DIR, exist_ok=True)
os.makedirs(Config.CKPT_DIR, exist_ok=True)
torch.manual_seed(Config.SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: ACTIVATION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def load_vision_tower():
    """Load the CLIP ViT-L/14-336 vision encoder."""
    if Config.LOAD_FROM_LLAVA:
        print(f"Loading vision tower from full LLaVA model: {Config.LLAVA_MODEL_ID}")
        from transformers import LlavaNextForConditionalGeneration
        full_model = LlavaNextForConditionalGeneration.from_pretrained(
            Config.LLAVA_MODEL_ID,
            torch_dtype=Config.DTYPE,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        vision_model = full_model.vision_tower
        del full_model
        gc.collect()
        vision_model = vision_model.to(Config.DEVICE)
    else:
        print(f"Loading standalone vision model: {Config.VISION_MODEL_ID}")
        print(f"  (This is the same CLIP ViT-L/14-336 used inside {Config.LLAVA_MODEL_ID})")
        from transformers import CLIPVisionModel
        vision_model = CLIPVisionModel.from_pretrained(
            Config.VISION_MODEL_ID,
            torch_dtype=Config.DTYPE,
        ).to(Config.DEVICE)

    vision_model.eval()
    return vision_model


def extract_activations():
    """
    Pass images through the vision encoder and save activations at target
    layers to disk.  Each layer gets its own set of chunk files.

    Output files:  {ACT_DIR}/layer_{L}/chunk_{NNNN}.pt
    """
    from transformers import CLIPImageProcessor
    from datasets import load_dataset

    print("\n" + "=" * 70)
    print("PHASE 1: Activation Extraction")
    print("=" * 70)

    processor = CLIPImageProcessor.from_pretrained(Config.VISION_MODEL_ID)
    model = load_vision_tower()

    # hidden_states index for each target layer
    hs_indices = {L: L + 1 for L in Config.TARGET_LAYERS}
    print(f"Target layers (0-idx): {Config.TARGET_LAYERS}")
    print(f"hidden_states indices:  {list(hs_indices.values())}")

    # Create per-layer output directories
    for L in Config.TARGET_LAYERS:
        os.makedirs(os.path.join(Config.ACT_DIR, f"layer_{L}"), exist_ok=True)

    # Stream ImageNet
    dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)

    buffers = {L: [] for L in Config.TARGET_LAYERS}
    chunk_idx = {L: 0 for L in Config.TARGET_LAYERS}
    zebra_count = 0
    other_count = 0
    max_other = Config.MAX_IMAGES - Config.MAX_ZEBRAS

    def flush_buffer(layer):
        if not buffers[layer]:
            return
        chunk_tensor = torch.cat(buffers[layer], dim=0)
        path = os.path.join(Config.ACT_DIR, f"layer_{layer}",
                            f"chunk_{chunk_idx[layer]:04d}.pt")
        torch.save(chunk_tensor, path)
        print(f"  Saved {path}  ({chunk_tensor.shape[0]:,} vectors)")
        buffers[layer] = []
        chunk_idx[layer] += 1

    with torch.no_grad():
        pbar = tqdm(dataset, desc="Extracting activations")
        for sample in pbar:
            # Check limits
            if zebra_count >= Config.MAX_ZEBRAS and other_count >= max_other:
                break

            is_zebra = (sample["label"] == Config.ZEBRA_LABEL)
            if is_zebra and zebra_count >= Config.MAX_ZEBRAS:
                continue
            if not is_zebra and other_count >= max_other:
                continue

            img = sample["image"].convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(Config.DEVICE, dtype=Config.DTYPE)
                      for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)

            # Extract activations at each target layer
            for L in Config.TARGET_LAYERS:
                hs = outputs.hidden_states[hs_indices[L]]
                # Drop CLS token (index 0), keep only patch tokens
                patch_acts = hs[0, 1:, :]       # shape: [576, 1024]
                buffers[L].append(patch_acts.cpu().to(torch.float32))

            if is_zebra:
                zebra_count += 1
            else:
                other_count += 1

            total = zebra_count + other_count
            pbar.set_postfix(total=total, zebra=zebra_count, other=other_count)

            # Flush buffers if they reach chunk size
            for L in Config.TARGET_LAYERS:
                buf_size = sum(b.shape[0] for b in buffers[L])
                if buf_size >= Config.CHUNK_SIZE:
                    flush_buffer(L)

    # Flush remaining
    for L in Config.TARGET_LAYERS:
        flush_buffer(L)

    summary = {
        "total_images": zebra_count + other_count,
        "zebra_images": zebra_count,
        "other_images": other_count,
        "patches_per_image": 576,
        "target_layers": Config.TARGET_LAYERS,
        "vision_model": Config.VISION_MODEL_ID,
        "llava_model": Config.LLAVA_MODEL_ID,
    }
    with open(os.path.join(Config.ACT_DIR, "extraction_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nExtraction complete. {summary['total_images']:,} images processed.")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: COMPUTE PREPROCESSING STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stats():
    """
    Compute per-layer mean activation and mean L2 norm.
    These are used to preprocess activations before SAE training:
        x_norm = (x - mean) / mean_norm

    Saves:  {ACT_DIR}/layer_{L}/stats.pt  containing {'mean', 'mean_norm'}
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Computing Preprocessing Statistics")
    print("=" * 70)

    for L in Config.TARGET_LAYERS:
        layer_dir = os.path.join(Config.ACT_DIR, f"layer_{L}")
        chunks = sorted(glob.glob(os.path.join(layer_dir, "chunk_*.pt")))
        if not chunks:
            print(f"  Layer {L}: No chunks found, skipping.")
            continue

        print(f"\n  Layer {L}: Computing stats across {len(chunks)} chunks ...")

        # Running sums for mean
        running_sum = torch.zeros(Config.D_MODEL, dtype=torch.float64)
        running_norm_sum = 0.0
        total_vectors = 0

        for cpath in tqdm(chunks, desc=f"  Layer {L} — pass 1 (mean)"):
            data = torch.load(cpath, map_location="cpu").to(torch.float64)
            running_sum += data.sum(dim=0)
            total_vectors += data.shape[0]

        mean = (running_sum / total_vectors).to(torch.float32)

        # Second pass: compute mean L2 norm of centered activations
        for cpath in tqdm(chunks, desc=f"  Layer {L} — pass 2 (norm)"):
            data = torch.load(cpath, map_location="cpu").to(torch.float32)
            centered = data - mean.unsqueeze(0)
            norms = centered.norm(dim=-1)          # [N]
            running_norm_sum += norms.sum().item()

        mean_norm = running_norm_sum / total_vectors

        stats = {"mean": mean, "mean_norm": torch.tensor(mean_norm)}
        stats_path = os.path.join(layer_dir, "stats.pt")
        torch.save(stats, stats_path)

        print(f"  Layer {L} stats:")
        print(f"    Total vectors   : {total_vectors:,}")
        print(f"    Mean vector norm: {mean.norm().item():.4f}")
        print(f"    Mean L2 norm    : {mean_norm:.4f}")
        print(f"    Saved to        : {stats_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: SAE ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

class TopKSAE(nn.Module):
    """
    TopK Sparse Autoencoder with improved initialisation.

    Key improvements:
    - W_enc initialised as W_dec.T  (prevents dead latents)
    - Decoder columns normalised to unit norm
    - Tracks feature activation counts for dead-feature detection
    """
    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae   = d_sae
        self.k       = k

        # ── Decoder (initialise FIRST so encoder can copy it) ──────────────
        self.W_dec = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # ── Encoder = transpose of decoder ─────────────────────────────────
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        self._init_weights()

    def _init_weights(self):
        # 1. Kaiming init for decoder
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))

        # 2. Normalise decoder columns to unit norm
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

        # 3. CRITICAL: encoder = transpose of decoder
        #    This prevents dead latents (OpenAI scaling-SAE paper)
        with torch.no_grad():
            self.W_enc.data = self.W_dec.data.T.clone()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to sparse latent space with TopK sparsity."""
        x_centered = x - self.b_dec
        pre_acts = F.linear(x_centered, self.W_enc, self.b_enc)

        # TopK: keep only top-k activations, apply ReLU, zero the rest
        topk_vals, topk_indices = torch.topk(pre_acts, self.k, dim=-1)
        h = torch.zeros_like(pre_acts)
        h.scatter_(-1, topk_indices, torch.relu(topk_vals))
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode from sparse latent space."""
        return F.linear(h, self.W_dec, self.b_dec)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            x_hat : reconstructed input
            h     : sparse latent activations
            loss  : MSE reconstruction loss
            info  : dict with L0 norm and active feature indices
        """
        h = self.encode(x)
        x_hat = self.decode(h)

        # MSE loss (sum over features, mean over batch)
        loss = (x_hat - x).pow(2).sum(dim=-1).mean()

        # ── Monitoring metrics ─────────────────────────────────────────────
        with torch.no_grad():
            active_mask = (h > 0)
            l0 = active_mask.float().sum(dim=-1).mean().item()
            # Which features activated in this batch (for dead-feature tracking)
            any_active = active_mask.any(dim=0)       # [d_sae] bool

        info = {"l0": l0, "any_active": any_active}
        return x_hat, h, loss, info

    @torch.no_grad()
    def normalize_decoder(self):
        """Normalise decoder columns to unit norm.  Call after each step."""
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3b: DATASET & TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

class ActivationDataset(Dataset):
    """Loads a single activation chunk, applies preprocessing."""
    def __init__(self, chunk_path, mean, mean_norm):
        raw = torch.load(chunk_path, map_location="cpu").to(torch.float32)
        # Preprocess: subtract mean, divide by mean L2 norm
        centered = raw - mean.unsqueeze(0)
        self.data = centered / mean_norm

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def train_sae_for_layer(layer: int):
    """Train one SAE on a single layer's activations."""
    layer_dir = os.path.join(Config.ACT_DIR, f"layer_{layer}")
    chunks = sorted(glob.glob(os.path.join(layer_dir, "chunk_*.pt")))
    if not chunks:
        raise ValueError(f"No activation chunks found for layer {layer}!")

    # Load preprocessing stats
    stats_path = os.path.join(layer_dir, "stats.pt")
    if not os.path.exists(stats_path):
        raise ValueError(f"Stats file not found: {stats_path}.  Run --phase stats first.")
    stats = torch.load(stats_path, map_location="cpu")
    mean      = stats["mean"]
    mean_norm = stats["mean_norm"].item()

    print(f"\n{'─' * 70}")
    print(f"  Training SAE for LAYER {layer}")
    print(f"{'─' * 70}")
    print(f"  Architecture : TopK SAE  ({Config.D_MODEL} → {Config.D_SAE}, k={Config.TOP_K})")
    print(f"  Chunks       : {len(chunks)}")
    print(f"  Batch size   : {Config.BATCH_SIZE}")
    print(f"  Learning rate: {Config.LEARNING_RATE}")
    print(f"  Preprocessing: subtract mean (norm={mean.norm():.4f}), "
          f"divide by {mean_norm:.4f}")

    # Build SAE
    sae = TopKSAE(Config.D_MODEL, Config.D_SAE, Config.TOP_K).to(Config.DEVICE)
    # Initialise b_dec to the dataset mean (gives the decoder a head start)
    with torch.no_grad():
        sae.b_dec.data = (mean / mean_norm).to(Config.DEVICE)

    optimizer = torch.optim.AdamW(
        sae.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    total_params = sum(p.numel() for p in sae.parameters())
    print(f"  SAE params   : {total_params:,}")

    # ── Dead-feature tracking ──────────────────────────────────────────────
    # Ring buffer: how many tokens since each feature last activated
    feature_last_active = torch.zeros(Config.D_SAE, dtype=torch.long,
                                      device=Config.DEVICE)
    global_token_count = 0

    # ── Resume from checkpoint ─────────────────────────────────────────────
    ckpt_path = os.path.join(Config.CKPT_DIR, f"sae_layer{layer}_latest.pt")
    start_epoch = 0
    start_chunk = 0
    if os.path.exists(ckpt_path):
        print(f"  Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=Config.DEVICE)
        sae.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        start_chunk = ckpt.get("chunk_idx", 0)
        global_token_count = ckpt.get("global_token_count", 0)

    # ── Training metrics log ───────────────────────────────────────────────
    metrics_log = []
    sae.train()

    for epoch in range(start_epoch, Config.EPOCHS):
        for c_idx in range(start_chunk, len(chunks)):
            chunk_file = chunks[c_idx]
            print(f"\n  Epoch {epoch + 1}/{Config.EPOCHS} | "
                  f"Chunk {c_idx + 1}/{len(chunks)} | "
                  f"{os.path.basename(chunk_file)}")

            dataset = ActivationDataset(chunk_file, mean, mean_norm)
            dataloader = DataLoader(
                dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                num_workers=2, pin_memory=True
            )

            batch_count = 0
            running_mse = 0.0
            running_l0  = 0.0
            running_ev  = 0.0

            pbar = tqdm(dataloader, desc=f"  L{layer}", leave=False)
            for batch in pbar:
                batch = batch.to(Config.DEVICE)
                batch_tokens = batch.shape[0]

                optimizer.zero_grad()
                x_hat, h, loss, info = sae(batch)
                loss.backward()
                optimizer.step()

                # Normalise decoder columns after each step
                sae.normalize_decoder()

                # ── Update dead-feature tracker ────────────────────────────
                with torch.no_grad():
                    global_token_count += batch_tokens
                    # Mark features that fired in this batch
                    fired = info["any_active"]      # [d_sae] bool
                    feature_last_active[fired] = global_token_count

                    # Explained variance
                    batch_var = batch.var(dim=-1).mean()
                    resid_var = (batch - x_hat).var(dim=-1).mean()
                    ev = (1 - resid_var / (batch_var + 1e-8)).item()

                # Accumulate for logging
                running_mse += loss.item()
                running_l0  += info["l0"]
                running_ev  += ev
                batch_count += 1

                if batch_count % Config.LOG_EVERY == 0:
                    avg_mse = running_mse / batch_count
                    avg_l0  = running_l0  / batch_count
                    avg_ev  = running_ev  / batch_count

                    # Dead features: not activated in the last DEAD_WINDOW tokens
                    tokens_since = global_token_count - feature_last_active
                    dead_count = (tokens_since > Config.DEAD_WINDOW).sum().item()
                    dead_pct = dead_count / Config.D_SAE * 100

                    pbar.set_postfix({
                        "MSE":  f"{avg_mse:.4f}",
                        "L0":   f"{avg_l0:.1f}",
                        "EV":   f"{avg_ev:.3f}",
                        "dead": f"{dead_count}/{Config.D_SAE} ({dead_pct:.1f}%)",
                    })

                    metrics_log.append({
                        "epoch": epoch,
                        "chunk": c_idx,
                        "batch": batch_count,
                        "global_tokens": global_token_count,
                        "mse": round(avg_mse, 6),
                        "l0":  round(avg_l0, 2),
                        "explained_variance": round(avg_ev, 4),
                        "dead_features": dead_count,
                        "dead_pct": round(dead_pct, 2),
                    })

            # ── End of chunk: log final metrics ────────────────────────────
            if batch_count > 0:
                avg_mse = running_mse / batch_count
                avg_l0  = running_l0  / batch_count
                avg_ev  = running_ev  / batch_count
                tokens_since = global_token_count - feature_last_active
                dead_count = (tokens_since > Config.DEAD_WINDOW).sum().item()
                dead_pct = dead_count / Config.D_SAE * 100
                print(f"  Chunk done │ MSE={avg_mse:.5f} │ L0={avg_l0:.1f}/{Config.TOP_K} │ "
                      f"EV={avg_ev:.4f} │ Dead={dead_count}/{Config.D_SAE} ({dead_pct:.1f}%)")

            # ── Save checkpoint after every chunk ──────────────────────────
            torch.save({
                "epoch": epoch,
                "chunk_idx": c_idx + 1,
                "model_state_dict": sae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_token_count": global_token_count,
                "feature_last_active": feature_last_active.cpu(),
                "loss": avg_mse if batch_count > 0 else None,
                "config": {
                    "d_model": Config.D_MODEL,
                    "d_sae": Config.D_SAE,
                    "top_k": Config.TOP_K,
                    "layer": layer,
                    "mean_norm": mean_norm,
                    "llava_model": Config.LLAVA_MODEL_ID,
                },
            }, ckpt_path)

        start_chunk = 0   # reset for next epoch

    # ── Save final model ───────────────────────────────────────────────────
    final_path = os.path.join(Config.CKPT_DIR, f"sae_layer{layer}_final.pt")
    torch.save(sae.state_dict(), final_path)
    print(f"\n  ✅ Layer {layer} SAE saved to {final_path}")

    # ── Save training metrics ──────────────────────────────────────────────
    metrics_path = os.path.join(Config.CKPT_DIR, f"sae_layer{layer}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"  📊 Metrics saved to {metrics_path}")

    return sae


def train_all_saes():
    """Train SAEs for all target layers sequentially."""
    print("\n" + "=" * 70)
    print("PHASE 3: SAE Training")
    print("=" * 70)

    for layer in Config.TARGET_LAYERS:
        train_sae_for_layer(layer)
        gc.collect()
        torch.cuda.empty_cache()

    # ── Print final summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — Summary")
    print("=" * 70)
    for layer in Config.TARGET_LAYERS:
        metrics_path = os.path.join(Config.CKPT_DIR,
                                    f"sae_layer{layer}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            if metrics:
                final = metrics[-1]
                print(f"  Layer {layer:2d} │ "
                      f"MSE={final['mse']:.5f} │ "
                      f"L0={final['l0']:.1f} │ "
                      f"EV={final['explained_variance']:.4f} │ "
                      f"Dead={final['dead_features']} ({final['dead_pct']:.1f}%)")

    print(f"\n  Checkpoints saved in: {Config.CKPT_DIR}/")
    print(f"  Files per layer:")
    print(f"    sae_layer{{L}}_final.pt    — model weights (for inference)")
    print(f"    sae_layer{{L}}_latest.pt   — full checkpoint (for resuming)")
    print(f"    sae_layer{{L}}_metrics.json — training curves")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train TopK SAEs on LLaVA-NeXT vision tower activations"
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["extract", "stats", "train", "all"],
        help="Which phase to run (default: all)"
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Override target layers (default: 12 17 22)"
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Override max images for extraction"
    )
    parser.add_argument(
        "--load-from-llava", action="store_true",
        help="Extract vision tower from full LLaVA model instead of standalone CLIP"
    )
    args = parser.parse_args()

    if args.layers:
        Config.TARGET_LAYERS = args.layers
    if args.max_images:
        Config.MAX_IMAGES = args.max_images
    if args.load_from_llava:
        Config.LOAD_FROM_LLAVA = True

    print("=" * 70)
    print("  TopK SAE Training for LLaVA-NeXT Vision Tower")
    print("=" * 70)
    print(f"  LLaVA model   : {Config.LLAVA_MODEL_ID}")
    print(f"  Vision model  : {Config.VISION_MODEL_ID}")
    print(f"  Target layers : {Config.TARGET_LAYERS}")
    print(f"  SAE dims      : {Config.D_MODEL} → {Config.D_SAE}  (k={Config.TOP_K})")
    print(f"  Device        : {Config.DEVICE}")
    print(f"  Activations   : {Config.ACT_DIR}")
    print(f"  Checkpoints   : {Config.CKPT_DIR}")
    print("=" * 70)

    if args.phase in ("extract", "all"):
        extract_activations()

    if args.phase in ("stats", "all"):
        compute_stats()

    if args.phase in ("train", "all"):
        train_all_saes()


if __name__ == "__main__":
    main()
