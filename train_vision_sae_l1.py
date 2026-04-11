"""
================================================================================
SAE Training for LLaVA-NeXT Llama3-8B Vision Tower (v3)
================================================================================

Trains ReLU+L1 Sparse Autoencoders on CLIP ViT-L/14-336 hidden states at
layer 22 (0-indexed) of the vision encoder inside
llava-hf/llama3-llava-next-8b-hf.

Changes over v2 (train_vision_sae_v2.py):
  1.  L1 loss       — Replaces TopK sparsity with L1 regularisation penalty
                       on the hidden activations: L = MSE + λ * ||h||₁
                       (matches the formulation in Pach et al. 2025, eq. 2-3)
  2.  ReLU encoder  — TopK gate removed; encoder now uses plain ReLU, letting
                       the L1 penalty determine sparsity naturally
  3.  Hyperparams   — λ (l1_coeff), expansion_factor, steps, batch_size, k(→L0
                       target) taken from monosemanticity_score.sh defaults:
                         expansion_factor = 4  (x4 in the shell script)
                         batch_size       = 4096
                         steps            = 110 000
                         l1_coeff         = 0.03  (maps to auxk_alpha in script)
                         decay_start      = 109 999  (LR cosine decay)
  4.  Skip extract  — If activation chunk files already exist for a layer,
                       Phase 1 is skipped and existing files are used directly
  5.  LR schedule   — Cosine decay from decay_start step to 0 at end, matching
                       the --decay_start flag in the reference shell script
  6.  Early stopping — Monitors MSE loss over a rolling window and stops if
                       improvement falls below min_delta for patience steps

Architecture per SAE:
  L1 SAE   (d_model=1024 → d_sae = d_model × expansion_factor)
  Encoder: h = ReLU( W_enc · (x - b_dec) + b_enc )
  Decoder: x̂ = W_dec · h + b_dec
  Loss:    L = MSE(x, x̂) + λ * ||h||₁

Usage:
  # Phase 1 — extract activations (GPU, ~2-3 hrs for 20k images)
  python train_vision_sae_v3.py --phase extract

  # Phase 2 — compute preprocessing stats (CPU OK, ~5 min)
  python train_vision_sae_v3.py --phase stats

  # Phase 3 — train SAEs (GPU, ~1-2 hrs per layer)
  python train_vision_sae_v3.py --phase train

  # All three phases sequentially
  python train_vision_sae_v3.py --phase all
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
    VISION_MODEL_ID = "openai/clip-vit-large-patch14-336"
    LOAD_FROM_LLAVA = False

    D_MODEL = 1024                       # CLIP ViT-L hidden dim

    # ── Layers to train SAEs on ────────────────────────────────────────────────
    TARGET_LAYERS = [22]

    # ── SAE architecture ───────────────────────────────────────────────────────
    # Expansion factor from shell script (--expansion_factor 4 default used)
    EXPANSION_FACTOR = 4
    D_SAE            = D_MODEL * EXPANSION_FACTOR   # 4 096

    # ── L1 sparsity ────────────────────────────────────────────────────────────
    # λ in L = MSE + λ * ||h||₁
    # Taken from --auxk_alpha 0.03 in monosemanticity_score.sh
    L1_COEFF = 0.03

    # ── Activation extraction ──────────────────────────────────────────────────
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE        = torch.float16 if torch.cuda.is_available() else torch.float32
    MAX_IMAGES   = 20_000
    MAX_ZEBRAS   = 1_000
    ZEBRA_LABEL  = 340
    CHUNK_SIZE   = 500_000
    ACT_DIR      = "./activations_v3"

    # ── SAE training ───────────────────────────────────────────────────────────
    BATCH_SIZE    = 4096           # --batch_size 4096 in shell script
    # Total optimisation steps from --steps 110000
    TOTAL_STEPS   = 110_000
    # LR cosine decay starts at this step (--decay_start 109999)
    DECAY_START   = 109_999
    # Learning rate formula from paper: 16 / (125 * sqrt(d_sae))
    # We compute it dynamically in train_sae_for_layer(); stored here as fallback
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY  = 0.0
    CKPT_DIR      = "./sae_checkpoints_v3"
    LOG_EVERY     = 50
    # Dead-feature window (in tokens)
    DEAD_WINDOW   = 50_000
    SEED = 42

    # ── Early Stopping ─────────────────────────────────────────────────────────
    EARLY_STOPPING       = True
    # Stop if no improvement for this many consecutive steps
    EARLY_STOP_PATIENCE  = 5_000
    # Minimum MSE improvement to count as meaningful progress
    EARLY_STOP_MIN_DELTA = 1e-6
    # Do not check early stopping before this many steps (let loss settle first)
    EARLY_STOP_WARMUP    = 10_000


# Create directories
os.makedirs(Config.ACT_DIR, exist_ok=True)
os.makedirs(Config.CKPT_DIR, exist_ok=True)
torch.manual_seed(Config.SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# EARLY STOPPING
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyStopper:
    """
    Monitors MSE loss step-by-step and signals early stopping when
    improvement drops below `min_delta` for `patience` consecutive steps.

    Args:
        patience  (int)   : Number of steps without meaningful improvement
                            before stopping.
        min_delta (float) : Minimum decrease in MSE to count as improvement.
        warmup    (int)   : Number of initial steps to skip before monitoring
                            begins (loss typically drops rapidly at the start).

    Usage:
        stopper = EarlyStopper(patience=5000, min_delta=1e-6, warmup=10000)
        ...
        if stopper.step(current_mse, global_step):
            break   # convergence detected — stop training
    """

    def __init__(self, patience: int, min_delta: float, warmup: int):
        self.patience   = patience
        self.min_delta  = min_delta
        self.warmup     = warmup
        self.best_loss  = float("inf")
        self.steps_without_improvement = 0

    def step(self, current_loss: float, global_step: int) -> bool:
        """
        Call once per training step with the current MSE.

        Returns:
            True  → stop training now (converged)
            False → keep going
        """
        # Skip monitoring during warm-up phase
        if global_step < self.warmup:
            return False

        improvement = self.best_loss - current_loss

        if improvement > self.min_delta:
            # Meaningful improvement — reset counter and update best
            self.best_loss = current_loss
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        if self.steps_without_improvement >= self.patience:
            print(
                f"\n  ⏹  Early stopping triggered at step {global_step}.\n"
                f"     No improvement > {self.min_delta} for "
                f"{self.patience} consecutive steps.\n"
                f"     Best MSE so far: {self.best_loss:.6f}"
            )
            return True

        return False

    @property
    def status(self) -> str:
        """Human-readable status string for logging."""
        return (
            f"best={self.best_loss:.6f}, "
            f"stale={self.steps_without_improvement}/{self.patience}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: ACTIVATION EXTRACTION  (skipped if chunks already exist)
# ═══════════════════════════════════════════════════════════════════════════════

def activations_exist(layer: int) -> bool:
    """Return True if at least one chunk file already exists for this layer."""
    layer_dir = os.path.join(Config.ACT_DIR, f"layer_{layer}")
    chunks = glob.glob(os.path.join(layer_dir, "chunk_*.pt"))
    return len(chunks) > 0


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
    layers to disk.  Skips extraction entirely for any layer whose chunk
    files already exist on disk.

    Output files:  {ACT_DIR}/layer_{L}/chunk_{NNNN}.pt
    """
    from transformers import CLIPImageProcessor
    from datasets import load_dataset

    print("\n" + "=" * 70)
    print("PHASE 1: Activation Extraction")
    print("=" * 70)

    # ── Check which layers actually need extraction ────────────────────────
    layers_to_extract = []
    for L in Config.TARGET_LAYERS:
        if activations_exist(L):
            print(f"  Layer {L}: Activation chunks already found — SKIPPING extraction.")
        else:
            layers_to_extract.append(L)

    if not layers_to_extract:
        print("  All layers already have activations. Nothing to extract.")
        return

    print(f"  Layers requiring extraction: {layers_to_extract}")

    processor = CLIPImageProcessor.from_pretrained(Config.VISION_MODEL_ID)
    model = load_vision_tower()

    hs_indices = {L: L + 1 for L in layers_to_extract}
    print(f"  hidden_states indices: {list(hs_indices.values())}")

    for L in layers_to_extract:
        os.makedirs(os.path.join(Config.ACT_DIR, f"layer_{L}"), exist_ok=True)

    dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)

    buffers   = {L: [] for L in layers_to_extract}
    chunk_idx = {L: 0  for L in layers_to_extract}
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

            for L in layers_to_extract:
                hs = outputs.hidden_states[hs_indices[L]]
                patch_acts = hs[0, 1:, :]       # drop CLS; shape [576, 1024]
                buffers[L].append(patch_acts.cpu().to(torch.float32))

            if is_zebra:
                zebra_count += 1
            else:
                other_count += 1

            total = zebra_count + other_count
            pbar.set_postfix(total=total, zebra=zebra_count, other=other_count)

            for L in layers_to_extract:
                buf_size = sum(b.shape[0] for b in buffers[L])
                if buf_size >= Config.CHUNK_SIZE:
                    flush_buffer(L)

    for L in layers_to_extract:
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
    Preprocessing applied before SAE training:
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

        stats_path = os.path.join(layer_dir, "stats.pt")
        if os.path.exists(stats_path):
            print(f"  Layer {L}: Stats already exist at {stats_path} — skipping recomputation.")
            continue

        print(f"\n  Layer {L}: Computing stats across {len(chunks)} chunks ...")

        running_sum   = torch.zeros(Config.D_MODEL, dtype=torch.float64)
        running_norm_sum = 0.0
        total_vectors = 0

        for cpath in tqdm(chunks, desc=f"  Layer {L} — pass 1 (mean)"):
            data = torch.load(cpath, map_location="cpu").to(torch.float64)
            running_sum += data.sum(dim=0)
            total_vectors += data.shape[0]

        mean = (running_sum / total_vectors).to(torch.float32)

        for cpath in tqdm(chunks, desc=f"  Layer {L} — pass 2 (norm)"):
            data = torch.load(cpath, map_location="cpu").to(torch.float32)
            centered = data - mean.unsqueeze(0)
            norms = centered.norm(dim=-1)
            running_norm_sum += norms.sum().item()

        mean_norm = running_norm_sum / total_vectors

        stats = {"mean": mean, "mean_norm": torch.tensor(mean_norm)}
        torch.save(stats, stats_path)

        print(f"  Layer {L} stats:")
        print(f"    Total vectors   : {total_vectors:,}")
        print(f"    Mean vector norm: {mean.norm().item():.4f}")
        print(f"    Mean L2 norm    : {mean_norm:.4f}")
        print(f"    Saved to        : {stats_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: SAE ARCHITECTURE  — L1 / ReLU variant
# ═══════════════════════════════════════════════════════════════════════════════

class L1SAE(nn.Module):
    """
    Sparse Autoencoder with ReLU activation and L1 sparsity penalty.

    Architecture (Pach et al. 2025, eq. 1-3):
        h    = ReLU( W_enc^T (v - b) + b_enc )   # encoder
        v̂    = W_dec^T h + b                      # decoder
        Loss = ||v - v̂||² + λ ||h||₁

    The shared bias b (b_dec) is subtracted from the encoder input and added
    back in the decoder, following the SAE formulation in the paper.

    Initialisation:
        W_dec  ~ Kaiming uniform, columns normalised to unit norm
        W_enc  = W_dec.T  (prevents dead latents, OpenAI scaling-SAE paper)
        b_dec  = dataset mean / mean_norm  (set externally after construction)
    """

    def __init__(self, d_model: int, d_sae: int, l1_coeff: float):
        super().__init__()
        self.d_model   = d_model
        self.d_sae     = d_sae
        self.l1_coeff  = l1_coeff

        # ── Decoder (initialise first so encoder can mirror it) ────────────
        self.W_dec = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))   # shared bias b

        # ── Encoder ────────────────────────────────────────────────────────
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        self._init_weights()

    def _init_weights(self):
        # 1. Kaiming init for decoder columns
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        # 2. Normalise decoder columns to unit norm
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)
        # 3. Encoder = transpose of decoder (reduces dead latents)
        with torch.no_grad():
            self.W_enc.data = self.W_dec.data.T.clone()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to sparse latent space via ReLU (no TopK gate)."""
        x_centered = x - self.b_dec           # subtract shared bias
        pre_acts   = F.linear(x_centered, self.W_enc, self.b_enc)
        return F.relu(pre_acts)               # h ∈ [0, ∞)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode from sparse latent space, add shared bias back."""
        return F.linear(h, self.W_dec, self.b_dec)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            x_hat  : reconstructed input
            h      : sparse latent activations  (ReLU output)
            loss   : MSE + λ * L1
            info   : dict with component losses and monitoring metrics
        """
        h     = self.encode(x)
        x_hat = self.decode(h)

        # ── Reconstruction loss (MSE, sum over features, mean over batch) ─
        mse_loss = (x_hat - x).pow(2).sum(dim=-1).mean()

        # ── L1 sparsity penalty (mean over batch, sum over latents) ───────
        l1_loss = h.abs().sum(dim=-1).mean()

        loss = mse_loss + self.l1_coeff * l1_loss

        # ── Monitoring ────────────────────────────────────────────────────
        with torch.no_grad():
            active_mask = (h > 0)
            l0          = active_mask.float().sum(dim=-1).mean().item()
            any_active  = active_mask.any(dim=0)   # [d_sae] bool

        info = {
            "l0":        l0,
            "mse":       mse_loss.item(),
            "l1":        l1_loss.item(),
            "any_active": any_active,
        }
        return x_hat, h, loss, info

    @torch.no_grad()
    def normalize_decoder(self):
        """Normalise decoder columns to unit norm after each optimiser step."""
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3b: DATASET & TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

class ActivationDataset(Dataset):
    """Loads a single activation chunk and applies mean/norm preprocessing."""
    def __init__(self, chunk_path: str, mean: torch.Tensor, mean_norm: float):
        raw     = torch.load(chunk_path, map_location="cpu").to(torch.float32)
        centered = raw - mean.unsqueeze(0)
        self.data = centered / mean_norm

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def cosine_lr(optimizer, step: int, total_steps: int,
              decay_start: int, base_lr: float):
    """
    Apply cosine LR decay starting at `decay_start` step.
    Before decay_start: constant base_lr.
    After: cosine from base_lr → 0.
    """
    if step < decay_start:
        lr = base_lr
    else:
        progress = (step - decay_start) / max(1, total_steps - decay_start)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_sae_for_layer(layer: int):
    """Train one L1-ReLU SAE on a single layer's activations."""
    layer_dir = os.path.join(Config.ACT_DIR, f"layer_{layer}")
    chunks = sorted(glob.glob(os.path.join(layer_dir, "chunk_*.pt")))
    if not chunks:
        raise ValueError(f"No activation chunks found for layer {layer}!")

    # Load preprocessing stats
    stats_path = os.path.join(layer_dir, "stats.pt")
    if not os.path.exists(stats_path):
        raise ValueError(f"Stats not found: {stats_path}. Run --phase stats first.")
    stats     = torch.load(stats_path, map_location="cpu")
    mean      = stats["mean"]
    mean_norm = stats["mean_norm"].item()

    # Learning rate from paper formula: 16 / (125 * sqrt(d_sae))
    lr = 16.0 / (125.0 * math.sqrt(Config.D_SAE))
    print(f"\n{'─' * 70}")
    print(f"  Training L1-ReLU SAE for LAYER {layer}")
    print(f"{'─' * 70}")
    print(f"  Architecture   : L1-ReLU SAE  ({Config.D_MODEL} → {Config.D_SAE})")
    print(f"  L1 coefficient : λ = {Config.L1_COEFF}")
    print(f"  Chunks         : {len(chunks)}")
    print(f"  Total steps    : {Config.TOTAL_STEPS}")
    print(f"  Batch size     : {Config.BATCH_SIZE}")
    print(f"  Learning rate  : {lr:.6f}  (decay from step {Config.DECAY_START})")
    print(f"  Preprocessing  : subtract mean (norm={mean.norm():.4f}), "
          f"divide by {mean_norm:.4f}")
    if Config.EARLY_STOPPING:
        print(f"  Early stopping : patience={Config.EARLY_STOP_PATIENCE} steps, "
              f"min_delta={Config.EARLY_STOP_MIN_DELTA}, "
              f"warmup={Config.EARLY_STOP_WARMUP} steps")
    else:
        print(f"  Early stopping : DISABLED")

    # Build SAE
    sae = L1SAE(Config.D_MODEL, Config.D_SAE, Config.L1_COEFF).to(Config.DEVICE)
    # Initialise b_dec to dataset mean (gives decoder a head start)
    with torch.no_grad():
        sae.b_dec.data = (mean / mean_norm).to(Config.DEVICE)

    optimizer = torch.optim.Adam(
        sae.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY
    )

    total_params = sum(p.numel() for p in sae.parameters())
    print(f"  SAE params     : {total_params:,}")

    # ── Early stopper ──────────────────────────────────────────────────────
    early_stopper = (
        EarlyStopper(
            patience  = Config.EARLY_STOP_PATIENCE,
            min_delta = Config.EARLY_STOP_MIN_DELTA,
            warmup    = Config.EARLY_STOP_WARMUP,
        )
        if Config.EARLY_STOPPING
        else None
    )

    # ── Dead-feature tracking ──────────────────────────────────────────────
    feature_last_active = torch.zeros(Config.D_SAE, dtype=torch.long,
                                      device=Config.DEVICE)
    global_token_count = 0
    global_step        = 0

    # ── Resume from checkpoint ─────────────────────────────────────────────
    ckpt_path  = os.path.join(Config.CKPT_DIR, f"sae_layer{layer}_latest.pt")
    start_chunk = 0
    if os.path.exists(ckpt_path):
        print(f"  Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=Config.DEVICE)
        sae.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_chunk        = ckpt.get("chunk_idx", 0)
        global_token_count = ckpt.get("global_token_count", 0)
        global_step        = ckpt.get("global_step", 0)
        # Restore early stopper state if available
        if early_stopper is not None and "early_stopper_state" in ckpt:
            es_state = ckpt["early_stopper_state"]
            early_stopper.best_loss = es_state.get("best_loss", float("inf"))
            early_stopper.steps_without_improvement = es_state.get(
                "steps_without_improvement", 0
            )
            print(f"  Early stopper restored: {early_stopper.status}")

    # ── Training metrics log ───────────────────────────────────────────────
    metrics_log = []
    early_stopped = False
    sae.train()

    # We cycle through chunks repeatedly until TOTAL_STEPS is reached
    chunk_order = list(range(len(chunks)))
    c_pos       = start_chunk % len(chunks)   # position in cycling order
    done        = (global_step >= Config.TOTAL_STEPS)

    pbar_global = tqdm(total=Config.TOTAL_STEPS, initial=global_step,
                       desc=f"  Layer {layer} steps")

    while not done:
        chunk_file = chunks[chunk_order[c_pos]]

        dataset    = ActivationDataset(chunk_file, mean, mean_norm)
        dataloader = DataLoader(
            dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=True
        )

        batch_count  = 0
        running_loss = 0.0
        running_mse  = 0.0
        running_l1   = 0.0
        running_l0   = 0.0
        running_ev   = 0.0

        for batch in dataloader:
            if global_step >= Config.TOTAL_STEPS:
                done = True
                break

            batch        = batch.to(Config.DEVICE)
            batch_tokens = batch.shape[0]

            # ── LR schedule ────────────────────────────────────────────────
            current_lr = cosine_lr(optimizer, global_step, Config.TOTAL_STEPS,
                                   Config.DECAY_START, lr)

            optimizer.zero_grad()
            x_hat, h, loss, info = sae(batch)
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

            # ── Early stopping check ───────────────────────────────────────
            if early_stopper is not None and early_stopper.step(info["mse"], global_step):
                done          = True
                early_stopped = True
                break

            # ── Dead-feature update ────────────────────────────────────────
            with torch.no_grad():
                global_token_count += batch_tokens
                global_step        += 1
                fired = info["any_active"]
                feature_last_active[fired] = global_token_count

                # Explained variance
                batch_var = batch.var(dim=-1).mean()
                resid_var = (batch - x_hat).var(dim=-1).mean()
                ev = (1 - resid_var / (batch_var + 1e-8)).item()

            running_loss += loss.item()
            running_mse  += info["mse"]
            running_l1   += info["l1"]
            running_l0   += info["l0"]
            running_ev   += ev
            batch_count  += 1

            pbar_global.update(1)

            if batch_count % Config.LOG_EVERY == 0:
                avg_loss = running_loss / batch_count
                avg_mse  = running_mse  / batch_count
                avg_l1   = running_l1   / batch_count
                avg_l0   = running_l0   / batch_count
                avg_ev   = running_ev   / batch_count

                tokens_since = global_token_count - feature_last_active
                dead_count   = (tokens_since > Config.DEAD_WINDOW).sum().item()
                dead_pct     = dead_count / Config.D_SAE * 100

                postfix = {
                    "Loss": f"{avg_loss:.4f}",
                    "MSE":  f"{avg_mse:.4f}",
                    "L1":   f"{avg_l1:.3f}",
                    "L0":   f"{avg_l0:.1f}",
                    "EV":   f"{avg_ev:.3f}",
                    "dead": f"{dead_count} ({dead_pct:.1f}%)",
                    "lr":   f"{current_lr:.2e}",
                }
                if early_stopper is not None:
                    postfix["ES"] = (
                        f"{early_stopper.steps_without_improvement}"
                        f"/{Config.EARLY_STOP_PATIENCE}"
                    )
                pbar_global.set_postfix(postfix)

                metrics_log.append({
                    "step":          global_step,
                    "global_tokens": global_token_count,
                    "loss":          round(avg_loss, 6),
                    "mse":           round(avg_mse,  6),
                    "l1":            round(avg_l1,   6),
                    "l0":            round(avg_l0,   2),
                    "explained_variance": round(avg_ev, 4),
                    "dead_features": dead_count,
                    "dead_pct":      round(dead_pct, 2),
                    "lr":            round(current_lr, 8),
                    # Early stopper state snapshot
                    "es_stale_steps": (
                        early_stopper.steps_without_improvement
                        if early_stopper else None
                    ),
                    "es_best_mse": (
                        round(early_stopper.best_loss, 6)
                        if early_stopper else None
                    ),
                })

        # ── End of chunk: save checkpoint ─────────────────────────────────
        if batch_count > 0:
            avg_mse  = running_mse  / batch_count
            avg_l0   = running_l0   / batch_count
            avg_ev   = running_ev   / batch_count
            tokens_since = global_token_count - feature_last_active
            dead_count   = (tokens_since > Config.DEAD_WINDOW).sum().item()
            dead_pct     = dead_count / Config.D_SAE * 100
            print(f"\n  Chunk {c_pos} done │ step={global_step} │ "
                  f"MSE={avg_mse:.5f} │ L0={avg_l0:.1f} │ "
                  f"EV={avg_ev:.4f} │ Dead={dead_count} ({dead_pct:.1f}%)"
                  + (f" │ ES stale={early_stopper.steps_without_improvement}"
                     if early_stopper else ""))

        # Persist early stopper state so training can be resumed correctly
        es_state = (
            {
                "best_loss": early_stopper.best_loss,
                "steps_without_improvement": early_stopper.steps_without_improvement,
            }
            if early_stopper else {}
        )

        torch.save({
            "chunk_idx":            (c_pos + 1) % len(chunks),
            "global_step":          global_step,
            "model_state_dict":     sae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_token_count":   global_token_count,
            "feature_last_active":  feature_last_active.cpu(),
            "loss": avg_mse if batch_count > 0 else None,
            "early_stopped":        early_stopped,
            "early_stopper_state":  es_state,
            "config": {
                "d_model":         Config.D_MODEL,
                "d_sae":           Config.D_SAE,
                "l1_coeff":        Config.L1_COEFF,
                "expansion_factor": Config.EXPANSION_FACTOR,
                "layer":           layer,
                "mean_norm":       mean_norm,
                "llava_model":     Config.LLAVA_MODEL_ID,
            },
        }, ckpt_path)

        if early_stopped:
            print(f"\n  ⏹  Training stopped early at step {global_step} "
                  f"(best MSE = {early_stopper.best_loss:.6f})")
            break

        # Advance to next chunk (cycling)
        c_pos = (c_pos + 1) % len(chunks)

    pbar_global.close()

    # ── Save final model ───────────────────────────────────────────────────
    final_path = os.path.join(Config.CKPT_DIR, f"sae_layer{layer}_final.pt")
    torch.save(sae.state_dict(), final_path)
    stop_reason = "early stopping" if early_stopped else "max steps reached"
    print(f"\n  ✅ Layer {layer} SAE saved to {final_path}  [{stop_reason}]")

    metrics_path = os.path.join(Config.CKPT_DIR, f"sae_layer{layer}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"  📊 Metrics saved to {metrics_path}")

    return sae


def train_all_saes():
    """Train L1-ReLU SAEs for all target layers sequentially."""
    print("\n" + "=" * 70)
    print("PHASE 3: SAE Training  (L1 + ReLU)")
    print("=" * 70)

    for layer in Config.TARGET_LAYERS:
        train_sae_for_layer(layer)
        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────
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
                      f"Loss={final['loss']:.5f} │ "
                      f"MSE={final['mse']:.5f} │ "
                      f"L1={final['l1']:.4f} │ "
                      f"L0={final['l0']:.1f} │ "
                      f"EV={final['explained_variance']:.4f} │ "
                      f"Dead={final['dead_features']} ({final['dead_pct']:.1f}%)")

    print(f"\n  Checkpoints in: {Config.CKPT_DIR}/")
    print(f"    sae_layer{{L}}_final.pt   — model weights (inference)")
    print(f"    sae_layer{{L}}_latest.pt  — full checkpoint (resuming)")
    print(f"    sae_layer{{L}}_metrics.json — training curves")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train L1-ReLU SAEs on LLaVA-NeXT vision tower activations"
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["extract", "stats", "train", "all"],
        help="Which phase to run (default: all)"
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Override target layers (default: [22])"
    )
    parser.add_argument(
        "--expansion-factor", type=int, default=None,
        help="Override expansion factor (default: 4)"
    )
    parser.add_argument(
        "--l1-coeff", type=float, default=None,
        help="Override L1 coefficient λ (default: 0.03)"
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Override max images for extraction"
    )
    parser.add_argument(
        "--total-steps", type=int, default=None,
        help="Override total training steps (default: 110000)"
    )
    parser.add_argument(
        "--load-from-llava", action="store_true",
        help="Extract vision tower from full LLaVA model instead of standalone CLIP"
    )
    # ── Early stopping CLI overrides ───────────────────────────────────────
    parser.add_argument(
        "--no-early-stopping", action="store_true",
        help="Disable early stopping (run for all TOTAL_STEPS unconditionally)"
    )
    parser.add_argument(
        "--es-patience", type=int, default=None,
        help=f"Early stopping patience in steps (default: {Config.EARLY_STOP_PATIENCE})"
    )
    parser.add_argument(
        "--es-min-delta", type=float, default=None,
        help=f"Early stopping min MSE improvement (default: {Config.EARLY_STOP_MIN_DELTA})"
    )
    parser.add_argument(
        "--es-warmup", type=int, default=None,
        help=f"Steps before early stopping activates (default: {Config.EARLY_STOP_WARMUP})"
    )
    args = parser.parse_args()

    if args.layers:
        Config.TARGET_LAYERS = args.layers
    if args.expansion_factor:
        Config.EXPANSION_FACTOR = args.expansion_factor
        Config.D_SAE = Config.D_MODEL * Config.EXPANSION_FACTOR
    if args.l1_coeff is not None:
        Config.L1_COEFF = args.l1_coeff
    if args.max_images:
        Config.MAX_IMAGES = args.max_images
    if args.total_steps:
        Config.TOTAL_STEPS = args.total_steps
    if args.load_from_llava:
        Config.LOAD_FROM_LLAVA = True
    if args.no_early_stopping:
        Config.EARLY_STOPPING = False
    if args.es_patience is not None:
        Config.EARLY_STOP_PATIENCE = args.es_patience
    if args.es_min_delta is not None:
        Config.EARLY_STOP_MIN_DELTA = args.es_min_delta
    if args.es_warmup is not None:
        Config.EARLY_STOP_WARMUP = args.es_warmup

    print("=" * 70)
    print("  L1-ReLU SAE Training for LLaVA-NeXT Vision Tower  (v3)")
    print("=" * 70)
    print(f"  LLaVA model      : {Config.LLAVA_MODEL_ID}")
    print(f"  Vision model     : {Config.VISION_MODEL_ID}")
    print(f"  Target layers    : {Config.TARGET_LAYERS}")
    print(f"  SAE dims         : {Config.D_MODEL} → {Config.D_SAE}  "
          f"(expansion_factor={Config.EXPANSION_FACTOR})")
    print(f"  Sparsity         : L1, λ = {Config.L1_COEFF}")
    print(f"  Total steps      : {Config.TOTAL_STEPS}  "
          f"(decay from step {Config.DECAY_START})")
    print(f"  Device           : {Config.DEVICE}")
    print(f"  Activations dir  : {Config.ACT_DIR}")
    print(f"  Checkpoints dir  : {Config.CKPT_DIR}")
    if Config.EARLY_STOPPING:
        print(f"  Early stopping   : patience={Config.EARLY_STOP_PATIENCE}, "
              f"min_delta={Config.EARLY_STOP_MIN_DELTA}, "
              f"warmup={Config.EARLY_STOP_WARMUP}")
    else:
        print(f"  Early stopping   : DISABLED")
    print("=" * 70)

    if args.phase in ("extract", "all"):
        extract_activations()

    if args.phase in ("stats", "all"):
        compute_stats()

    if args.phase in ("train", "all"):
        train_all_saes()


if __name__ == "__main__":
    main()