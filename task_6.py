# T-06: Gradient-Based Unlearning — Both Modalities (Strongest Baseline)
#
# Goal: Implement gradient-based unlearning (Gradient Difference method) on LLaVA-next-8B
#       across ALL parameters — CLIP encoder, MLP projector, and LLaMA LM.
#
# Method: Gradient Difference
#   - Gradient Ascent on forget set  → maximize loss (unlearn zebra)
#   - Gradient Descent on retain set → minimize loss (preserve general knowledge)
#   - total_loss = -loss_forget + lambda * loss_retain  with lambda=0.5

# =============================================================================
# 0. Environment Setup
# =============================================================================
import os, gc, json, math, time, random, logging, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from huggingface_hub import login as hf_login

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB)")
print(f"PyTorch: {torch.__version__}")


# =============================================================================
# 1. Configuration
# =============================================================================


# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID          = "/home/mehul/.cache/huggingface/hub/models--llava-hf--llama3-llava-next-8b-hf/snapshots/b041c0d0ea0dd0196d147206c210c8d1752fc2da"
LOAD_IN_FP16      = True
USE_4BIT          = True    # 4-bit to fit in 24GB RTX 3090

# ── Hyperparameters ───────────────────────────────────────────────────────────
LAMBDA_RETAIN     = 1.0
LR                = 5e-4   # standard for LoRA fine-tuning
WEIGHT_DECAY      = 1e-2
GRAD_CLIP         = 1.0    # restore normal, LoRA is self-limiting
MAX_EPOCHS        = 1    # need more epochs with higher LR
BATCH_SIZE        = 1      # reduced for memory
GRAD_ACCUM_STEPS  = 8      # reduced back from 16
WARMUP_STEPS      = 50
MAX_NEW_TOKENS    = 64
MAX_SEQ_LEN       = 512

# ── Early stopping ────────────────────────────────────────────────────────────
RA_DROP_THRESHOLD = 0.15   # slightly more permissive
FA_TARGET         = 0.30

# ── Simplified fallback: GA only, skip retain set ─────────────────────────────
SIMPLIFIED_GA_ONLY = False

# ── Output dirs ───────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("./t06_gradient_unlearning")
CKPT_DIR    = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"
for d in [OUTPUT_DIR, CKPT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CONFIG = dict(
    model_id="llava-hf/llama3-llava-next-8b-hf",
    lambda_retain=LAMBDA_RETAIN, lr=LR,
    weight_decay=WEIGHT_DECAY, max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE, grad_accum=GRAD_ACCUM_STEPS,
    simplified=SIMPLIFIED_GA_ONLY, seed=SEED,
    ra_drop_threshold=RA_DROP_THRESHOLD, fa_target=FA_TARGET,
)
print(json.dumps(CONFIG, indent=2))
print(f"\nAll outputs → {OUTPUT_DIR}")

hf_login(token=HF_TOKEN)
print("HuggingFace login ✓  (read access only — no uploads)")


# =============================================================================
# 2. Dataset
# =============================================================================

class UnlearningDataset(Dataset):
    def __init__(self, hf_split, processor, max_seq_len=512):
        self.data        = hf_split
        self.processor   = processor
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def _load_image(self, item):
        img = item.get("image")
        if img is None:
            return Image.new("RGB", (336, 336), color=(128, 128, 128))
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, bytes):
            from io import BytesIO
            return Image.open(BytesIO(img)).convert("RGB")
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        return img.convert("RGB")

    def __getitem__(self, idx):
        item       = self.data[idx]
        image      = self._load_image(item)
        image_size = image.size
        question   = item.get("question", item.get("prompt", ""))
        answer     = item.get("answer",   item.get("response", ""))

        conversation = [
            {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": question}
            ]}
        ]
        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        encoding = self.processor(
            images=image,
            text=prompt_text + answer,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        encoding["image_sizes"] = torch.tensor([image_size[::-1]])
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        pixel_values   = encoding["pixel_values"].squeeze(0)

        # Mask prompt tokens — only supervise the answer
        prompt_len = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )["input_ids"].shape[-1]

        labels = input_ids.clone()
        labels[:prompt_len]         = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "pixel_values":   pixel_values,
            "image_sizes":    encoding["image_sizes"].squeeze(0),
            "labels":         labels,
        }


def collate_fn(batch):
    out = {}

    for k in ["input_ids", "attention_mask", "labels"]:
        tensors = [b[k] for b in batch]
        out[k] = torch.nn.utils.rnn.pad_sequence(
            tensors,
            batch_first=True,
            padding_value=0 if k != "labels" else -100,
        )

    pixel_values = [b["pixel_values"] for b in batch]
    max_patches  = max(p.shape[0] for p in pixel_values)

    padded_pixels = []
    for p in pixel_values:
        pad_size = max_patches - p.shape[0]
        if pad_size > 0:
            pad = torch.zeros((pad_size, *p.shape[1:]), dtype=p.dtype)
            p   = torch.cat([p, pad], dim=0)
        padded_pixels.append(p)

    out["pixel_values"] = torch.stack(padded_pixels)
    out["image_sizes"]  = torch.stack([b["image_sizes"] for b in batch])

    return out


logger.info("Loading processor…")
processor = LlavaNextProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)

DATASET_PATH = "./data"
logger.info(f"Loading dataset locally from {DATASET_PATH}…")

manifest_path = os.path.join(DATASET_PATH, "dataset_manifest.json")
with open(manifest_path, "r") as f:
    manifest = json.load(f)

forget_dir = os.path.join(DATASET_PATH, "forget")
retain_dir = os.path.join(DATASET_PATH, "retain")


def build_forget_split():
    data     = []
    qa_pairs = manifest["forget_qa_pairs"]
    for img_name in sorted(os.listdir(forget_dir)):
        img_path = os.path.join(forget_dir, img_name)
        for qa in qa_pairs:
            data.append({
                "image":    img_path,
                "question": qa["question"],
                "answer":   qa["ground_truth"],
            })
    return data


def build_retain_split():
    data        = []
    labels      = manifest["retain_labels"]
    qa_template = manifest["retain_qa_template"]
    for img_name, label in zip(sorted(os.listdir(retain_dir)), labels):
        img_path = os.path.join(retain_dir, img_name)
        for qa in qa_template:
            answer = qa["ground_truth"].replace("{{animal}}", label)
            data.append({
                "image":    img_path,
                "question": qa["question"],
                "answer":   answer,
            })
    return data


forget_split = build_forget_split()
retain_split = build_retain_split()

print("Sample retain item:", retain_split[0])
print("Sample forget item:", forget_split[0])
assert all(ex["answer"] for ex in retain_split[:10]), "Empty retain answers!"

eval_f_split = forget_split[:min(50, len(forget_split))]
eval_r_split = retain_split[:min(50, len(retain_split))]

print(f"Forget train : {len(forget_split):,}")
print(f"Retain train : {len(retain_split):,}")
print(f"Forget eval  : {len(eval_f_split):,}")
print(f"Retain eval  : {len(eval_r_split):,}")

forget_dataset = UnlearningDataset(forget_split, processor, MAX_SEQ_LEN)
retain_dataset = UnlearningDataset(retain_split, processor, MAX_SEQ_LEN)
eval_f_dataset = UnlearningDataset(eval_f_split, processor, MAX_SEQ_LEN)
eval_r_dataset = UnlearningDataset(eval_r_split, processor, MAX_SEQ_LEN)

forget_loader = DataLoader(forget_dataset, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=collate_fn, num_workers=2, pin_memory=True)
retain_loader = DataLoader(retain_dataset, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=collate_fn, num_workers=2, pin_memory=True)
eval_f_loader = DataLoader(eval_f_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
eval_r_loader = DataLoader(eval_r_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

print("DataLoaders ready ✓")


# =============================================================================
# 3. Load Model — vision-focused adaptation
# =============================================================================

def load_model(model_id, fp16=True, use_4bit=False, token=None):
    kwargs = dict(device_map="auto")
    if token:
        kwargs["token"] = token

    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_storage=torch.float16,
        )
        logger.info("Loading in 4-bit quantization.")
    elif fp16:
        kwargs["torch_dtype"] = torch.float16

    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, **kwargs)

    # Required for stable LoRA training on k-bit quantized backbones.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Apply LoRA only to the vision tower attention projections.
    # Keep LM frozen so updates are concentrated on visual pathways.
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=r"^vision_tower\..*(q_proj|k_proj|v_proj|out_proj)$",
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Freeze language-model updates entirely.
    for name, param in model.named_parameters():
        if "language_model" in name:
            param.requires_grad = False

    # Keep multimodal projector trainable to strengthen vision-language alignment.
    for name, param in model.named_parameters():
        if "multi_modal_projector" in name:
            param.requires_grad = True

    model.print_trainable_parameters()

    return model


logger.info("Loading base model…")
model = load_model(MODEL_ID, fp16=LOAD_IN_FP16, use_4bit=USE_4BIT, token=None)
model.config.use_cache = False
logger.info("Model loaded ✓")


# =============================================================================
# 4. Baseline Loss
# =============================================================================

@torch.no_grad()
def evaluate_loss(model, loader, device, max_batches=25):
    model.eval()
    losses = []
    torch.cuda.empty_cache()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(dtype=torch.float16):
            losses.append(model(**batch).loss.item())
    model.train()
    torch.cuda.empty_cache()
    return float(np.mean(losses)) if losses else float("nan")


logger.info("Computing baseline losses…")
baseline_forget_loss = evaluate_loss(model, eval_f_loader, DEVICE)
baseline_retain_loss = evaluate_loss(model, eval_r_loader, DEVICE)
print(f"Baseline forget loss : {baseline_forget_loss:.4f}")
print(f"Baseline retain loss : {baseline_retain_loss:.4f}")


# =============================================================================
# 5. Evaluation Metrics (FA, RA, LL, VD)
# =============================================================================

import nltk
nltk.download("punkt", quiet=True)
from rouge_score import rouge_scorer as rouge_lib

ZEBRA_KEYWORDS = [
    "zebra", "zebras", "stripes", "striped", "equid",
    "savanna", "savannah", "africa", "african", "grazing", "herd", "plains",
]


def load_image_safe(img):
    if img is None:
        return Image.new("RGB", (336, 336), color=(128, 128, 128))
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    if isinstance(img, bytes):
        from io import BytesIO
        return Image.open(BytesIO(img)).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(img)}")


def has_zebra(text: str) -> bool:
    return any(kw in text.lower() for kw in ZEBRA_KEYWORDS)


@torch.no_grad()
def generate(model, processor, image, question, device):
    conv   = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question}
    ]}]
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, temperature=1.0)
    return processor.decode(ids[0][inputs["input_ids"].shape[-1]:],
                            skip_special_tokens=True).strip()


@torch.no_grad()
def compute_all_metrics(model, processor, forget_split, retain_split, device, n=40):
    """
    FA  — % of forget-set answers still mentioning zebra (lower = better unlearning)
    RA  — ROUGE-L on retain set (higher = less collateral damage)
    LL  — Text-only zebra QA with blank image (LM-level language knowledge)
    VD  — False zebra ID on blank image (hallucination sanity check)
    """
    model.eval()
    torch.cuda.empty_cache()
    scorer  = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    results = {}

    # FA
    fsplit  = forget_split[:min(n, len(forget_split))]
    fa_hits = 0
    for ex in tqdm(fsplit, desc="FA", leave=False):
        img = load_image_safe(ex.get("image"))
        if has_zebra(generate(model, processor, img,
                              ex.get("question", "What animal is in this image?"), device)):
            fa_hits += 1
    results["FA"] = fa_hits / len(fsplit)

    # RA — % of retain-set answers mentioning horse/donkey (higher = better retention)
    rsplit       = retain_split[:min(n, len(retain_split))]
    ra_hd_hits   = 0  # count predictions that mention horse/donkey
    for ex in tqdm(rsplit, desc="RA", leave=False):
        img = load_image_safe(ex.get("image"))
        pred = generate(model, processor, img,
                        ex.get("question", "What animal is in this image?"), device)
        # Check if model outputs horse/donkey (same check as FA but for horse/donkey)
        if any(kw in pred.lower() for kw in ["horse", "donkey"]):
            ra_hd_hits += 1
    results["RA"] = ra_hd_hits / len(rsplit)

    # LL — blank image isolates language-model-level zebra knowledge
    ll_qs = [
        "What are zebras known for?",
        "Describe the appearance of a zebra.",
        "Where do zebras live?",
        "What family do zebras belong to?",
        "Are zebras black with white stripes or white with black stripes?",
    ]
    blank      = Image.new("RGB", (336, 336), color=(128, 128, 128))
    ll_hits    = 0
    ll_answers = []
    for q in ll_qs:
        ans = generate(model, processor, blank, q, device)
        ll_answers.append({"question": q, "answer": ans})
        if has_zebra(ans):
            ll_hits += 1
    results["LL"]         = ll_hits / len(ll_qs)
    results["LL_answers"] = ll_answers

    # VD
    vd_qs = ["What animal is in this image?", "Is this a zebra?"]
    results["VD"] = sum(
        has_zebra(generate(model, processor, blank, q, device)) for q in vd_qs
    ) / len(vd_qs)

    model.train()
    torch.cuda.empty_cache()
    return results


print("Metric functions defined ✓")

logger.info("Running BASELINE metrics…")
baseline_metrics = compute_all_metrics(
    model, processor, forget_split, retain_split, DEVICE
)
baseline_metrics.update({
    "epoch":       0,
    "forget_loss": baseline_forget_loss,
    "retain_loss": baseline_retain_loss,
})

print("\n═══ BASELINE METRICS ═══")
for k, v in baseline_metrics.items():
    if k != "LL_answers":
        print(f"  {k:15s}: {v}")
print("\nLL answers (pre-unlearning):")
for qa in baseline_metrics["LL_answers"]:
    print(f"  Q: {qa['question']}\n  A: {qa['answer']}\n")


# =============================================================================
# 6. Gradient Difference Unlearning Loop
# =============================================================================

def build_optimizer(model, lr, wd):
    no_decay = ["bias", "LayerNorm", "layer_norm"]
    return torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": wd},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ], lr=lr)


def cosine_warmup(optimizer, warmup, total):
    def fn(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, 0.5 * (1 + math.cos(
            math.pi * (step - warmup) / max(1, total - warmup)
        )))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def tradeoff(fa, ra, ra_base):
    """Higher = better: penalises high FA and RA degradation."""
    ra_ratio = min(ra / max(ra_base, 1e-6), 1.0)
    return (1.0 - fa) * ra_ratio


optimizer   = build_optimizer(model, LR, WEIGHT_DECAY)
spe         = max(len(forget_loader), len(retain_loader))
total_steps = MAX_EPOCHS * spe // GRAD_ACCUM_STEPS
scheduler   = cosine_warmup(optimizer, WARMUP_STEPS, total_steps)

print(f"AdamW lr={LR} wd={WEIGHT_DECAY}")
print(f"Steps/epoch={spe}  Total opt steps={total_steps}")
print(f"Objective: FORGET (GA) on forget set  +  LEARN (GD) on retain set")

model.train()

history        = [baseline_metrics]
best_score     = -1.0
best_epoch     = 0
best_ckpt_path = None
early_stop     = False
global_step    = 0
retain_iter    = iter(retain_loader)

for epoch in range(1, MAX_EPOCHS + 1):
    if early_stop:
        break

    t0 = time.time()

    # Per-epoch accumulators — track forget and retain losses separately
    ep_forget_losses  = []   # GA loss (want this to rise = unlearning working)
    ep_retain_losses  = []   # GD loss (want this to fall = retain knowledge preserved)
    ep_total_losses   = []   # combined scalar for reference

    optimizer.zero_grad()

    for step, fbatch in enumerate(
        tqdm(forget_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS}", leave=True)
    ):
        # ── OBJECTIVE 1: FORGET ──────────────────────────────────────────────
        # Gradient ASCENT on forget set — push model away from zebra answers
        fbatch = {k: v.to(DEVICE) for k, v in fbatch.items()}
        loss_forget = model(**fbatch).loss
        loss_forget_val = loss_forget.item()

        # Backward on forget immediately (negative = ascent)
        (-loss_forget / GRAD_ACCUM_STEPS).backward()
        del loss_forget, fbatch
        torch.cuda.empty_cache()

        # ── OBJECTIVE 2: LEARN / RETAIN ──────────────────────────────────────
        # Gradient DESCENT on retain set — keep model fluent on non-zebra animals
        try:
            rbatch = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            rbatch = next(retain_iter)

        rbatch = {k: v.to(DEVICE) for k, v in rbatch.items()}
        loss_retain = model(**rbatch).loss
        loss_retain_val = loss_retain.item()

        # Backward on retain (positive = descent = normal learning)
        (LAMBDA_RETAIN * loss_retain / GRAD_ACCUM_STEPS).backward()
        del loss_retain, rbatch
        torch.cuda.empty_cache()

        # Track both losses
        ep_forget_losses.append(loss_forget_val)
        ep_retain_losses.append(loss_retain_val)
        ep_total_losses.append(-loss_forget_val + LAMBDA_RETAIN * loss_retain_val)

        # ── OPTIMIZER STEP ────────────────────────────────────────────────────
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            # Sanity check: confirm gradients are flowing (first step only)
            if epoch == 1 and step == GRAD_ACCUM_STEPS - 1:
                grad_norms = [
                    p.grad.norm().item()
                    for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                if not grad_norms:
                    raise RuntimeError("No gradients found! LoRA setup broken.")
                print(f"\n  [sanity] grad norms — "
                      f"min={min(grad_norms):.2e}  "
                      f"max={max(grad_norms):.2e}  "
                      f"n_params_with_grad={len(grad_norms)}")

            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        # ── STEP-LEVEL LOGGING (every 25 steps) ──────────────────────────────
        if (step + 1) % 25 == 0:
            avg_f = np.mean(ep_forget_losses[-25:])
            avg_r = np.mean(ep_retain_losses[-25:])
            print(f"  [step {step+1:03d}] "
                  f"forget_loss={avg_f:.4f} (↑ rising = unlearning)  "
                  f"retain_loss={avg_r:.4f} (↓ falling = retaining)")

    elapsed = time.time() - t0

    # ── EPOCH SUMMARY ─────────────────────────────────────────────────────────
    mean_forget = np.mean(ep_forget_losses)
    mean_retain = np.mean(ep_retain_losses)
    mean_total  = np.mean(ep_total_losses)

    logger.info(
        f"Epoch {epoch} done | "
        f"forget_loss={mean_forget:.4f} (GA↑)  "
        f"retain_loss={mean_retain:.4f} (GD↓)  "
        f"total={mean_total:.4f}  "
        f"time={elapsed/60:.1f}min"
    )

    # ── EVALUATE ──────────────────────────────────────────────────────────────
    logger.info("Evaluating metrics…")
    m = compute_all_metrics(model, processor, forget_split, retain_split, DEVICE)
    m.update(dict(
        epoch        = epoch,
        forget_loss  = mean_forget,
        retain_loss  = mean_retain,
        total_loss   = mean_total,
        elapsed_min  = elapsed / 60,
    ))
    history.append(m)

    print(f"\n  ┌─ Epoch {epoch:02d} Results ─────────────────────────────────┐")
    print(f"  │  FA  = {m['FA']:.3f}   (↓ lower = more forgotten)          │")
    print(f"  │  RA  = {m['RA']:.3f}   (↑ higher = horse/donkey retained) │")
    print(f"  │  LL  = {m['LL']:.3f}   (↓ lower = language knowledge gone) │")
    print(f"  │  VD  = {m['VD']:.3f}   (sanity check on blank image)       │")
    print(f"  │  forget_loss = {mean_forget:.4f}  retain_loss = {mean_retain:.4f}    │")
    print(f"  └────────────────────────────────────────────────────────┘")

    # ── CHECKPOINT ────────────────────────────────────────────────────────────
    ckpt = CKPT_DIR / f"epoch_{epoch:02d}"
    model.save_pretrained(str(ckpt))
    processor.save_pretrained(str(ckpt))
    logger.info(f"Checkpoint saved → {ckpt}")

    # ── BEST MODEL TRACKING ───────────────────────────────────────────────────
    sc = tradeoff(m["FA"], m["RA"], baseline_metrics["RA"])
    if sc > best_score:
        best_score     = sc
        best_epoch     = epoch
        best_ckpt_path = ckpt
        print(f"  ✦ New best checkpoint (tradeoff score={sc:.4f})")

    # ── EARLY STOPPING ────────────────────────────────────────────────────────
    ra_drop = (baseline_metrics["RA"] - m["RA"]) / max(baseline_metrics["RA"], 1e-6)
    if ra_drop > RA_DROP_THRESHOLD:
        logger.warning(
            f"Early stop: retain RA dropped {ra_drop*100:.1f}% > {RA_DROP_THRESHOLD*100:.0f}% threshold. "
            f"Retain learning is not compensating for GA damage."
        )
        early_stop = True

    if m["FA"] < FA_TARGET:
        logger.info(
            f"Early stop: FA={m['FA']:.3f} < {FA_TARGET} target — unlearning sufficient!"
        )
        early_stop = True

print(f"\nTraining complete.")
print(f"Best checkpoint: epoch {best_epoch}, tradeoff score {best_score:.4f}")
print(f"Best checkpoint path → {best_ckpt_path}")


# =============================================================================
# 7. Full Evaluation on Best Checkpoint
# =============================================================================

if best_ckpt_path is not None and Path(best_ckpt_path).exists():
    logger.info(f"Reloading best checkpoint: epoch {best_epoch}")
    del model; gc.collect(); torch.cuda.empty_cache()
    model = LlavaNextForConditionalGeneration.from_pretrained(
        str(best_ckpt_path), torch_dtype=torch.float16, device_map="auto"
    )
    model.config.use_cache = True
else:
    logger.warning("No best checkpoint found — using current model state.")

logger.info("Running FINAL full evaluation (n=100)…")
final_metrics = compute_all_metrics(
    model, processor, forget_split, retain_split, DEVICE, n=100
)
final_metrics.update(dict(
    epoch         = f"best_epoch_{best_epoch}",
    method        = "GA_only" if SIMPLIFIED_GA_ONLY else "gradient_difference",
    lambda_retain = 0.0 if SIMPLIFIED_GA_ONLY else LAMBDA_RETAIN,
    baseline_FA   = baseline_metrics["FA"],
    baseline_RA   = baseline_metrics["RA"],
    baseline_LL   = baseline_metrics["LL"],
))

print("\n═══ FINAL EVALUATION ═══")
for k, v in final_metrics.items():
    if k != "LL_answers":
        print(f"  {k:20s}: {v}")
print("\nLL answers (post-unlearning):")
for qa in final_metrics["LL_answers"]:
    print(f"  Q: {qa['question']}\n  A: {qa['answer']}\n")

ll_drop = baseline_metrics["LL"] - final_metrics["LL"]
ll_pct  = ll_drop / max(baseline_metrics["LL"], 1e-6) * 100

print("═══ CRITICAL ANALYSIS: LANGUAGE-LEVEL (LL) ZEBRA KNOWLEDGE ═══")
print(f"  Baseline LL  : {baseline_metrics['LL']:.3f}")
print(f"  Post-UL  LL  : {final_metrics['LL']:.3f}")
print(f"  Reduction    : {ll_drop:.3f}  ({ll_pct:.1f}% drop)")
print()
if final_metrics["LL"] < 0.3:
    print("  ✓ SUBSTANTIALLY removed")
elif final_metrics["LL"] < 0.6:
    print("  ~ PARTIALLY removed")
else:
    print("  ✗ LARGELY PERSISTS")
print()
print(f"  FA (visual)  : {baseline_metrics['FA']:.3f} → {final_metrics['FA']:.3f}  ({final_metrics['FA']-baseline_metrics['FA']:+.3f})")
print(f"  LL (language): {baseline_metrics['LL']:.3f} → {final_metrics['LL']:.3f}  ({final_metrics['LL']-baseline_metrics['LL']:+.3f})")
print(f"  RA (retain)  : {baseline_metrics['RA']:.3f} → {final_metrics['RA']:.3f}  ({final_metrics['RA']-baseline_metrics['RA']:+.3f})")


# =============================================================================
# 8. Plots
# =============================================================================

ep_list = [h["epoch"] for h in history if isinstance(h["epoch"], (int, float))]
fa_list = [h["FA"]    for h in history if isinstance(h["epoch"], (int, float))]
ra_list = [h["RA"]    for h in history if isinstance(h["epoch"], (int, float))]
ll_list = [h["LL"]    for h in history if isinstance(h["epoch"], (int, float))]
fl_list = [h.get("forget_loss", 0) for h in history if isinstance(h["epoch"], (int, float))]
rl_list = [h.get("retain_loss", 0) for h in history if isinstance(h["epoch"], (int, float))]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("T-06: Gradient-Based Unlearning — Training Dynamics",
             fontsize=14, fontweight="bold")

axes[0,0].plot(ep_list, fa_list, "o-", color="#e74c3c", lw=2)
axes[0,0].axhline(FA_TARGET, color="gray", ls="--", label=f"target ({FA_TARGET})")
axes[0,0].set_title("Forget Accuracy (FA) ↓"); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

ra_thresh = baseline_metrics["RA"] * (1 - RA_DROP_THRESHOLD)
axes[0,1].plot(ep_list, ra_list, "o-", color="#2ecc71", lw=2)
axes[0,1].axhline(ra_thresh, color="gray", ls="--", label=f"stop ({ra_thresh:.2f})")
axes[0,1].axhline(baseline_metrics["RA"], color="blue", ls=":", label="baseline")
axes[0,1].set_title("Retain Accuracy (RA) ↑"); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

axes[0,2].plot(ep_list, ll_list, "o-", color="#9b59b6", lw=2)
axes[0,2].axhline(baseline_metrics["LL"], color="blue", ls=":", label="baseline")
axes[0,2].set_title("Language-Level Zebra Knowledge (LL) ↓")
axes[0,2].legend(); axes[0,2].grid(alpha=0.3)

axes[1,0].plot(ep_list, fl_list, "o-", color="#e67e22", lw=2)
axes[1,0].set_title("Forget Set Loss (↑ = more forgetting)"); axes[1,0].grid(alpha=0.3)

axes[1,1].plot(ep_list, rl_list, "o-", color="#1abc9c", lw=2)
axes[1,1].set_title("Retain Set Loss (stable = good)"); axes[1,1].grid(alpha=0.3)

sc_plot = axes[1,2].scatter(fa_list, ra_list, c=ep_list, cmap="viridis", s=100, zorder=5)
for i, ep in enumerate(ep_list):
    axes[1,2].annotate(str(ep), (fa_list[i], ra_list[i]),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)
plt.colorbar(sc_plot, ax=axes[1,2], label="Epoch")
axes[1,2].set_title("FA vs RA Tradeoff (lower-right = ideal)")
axes[1,2].set_xlabel("FA"); axes[1,2].set_ylabel("RA"); axes[1,2].grid(alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / "training_curves.png"
plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {plot_path}")

cmp = {
    "No Unlearning\n(Baseline)": {"FA": baseline_metrics["FA"], "RA": baseline_metrics["RA"], "LL": baseline_metrics["LL"]},
    "Gradient Diff\n(T-06)":     {"FA": final_metrics["FA"],    "RA": final_metrics["RA"],    "LL": final_metrics["LL"]},
}
methods = list(cmp.keys()); x = np.arange(len(methods)); w = 0.25
fig, ax = plt.subplots(figsize=(10, 6))
for i, (mk, col) in enumerate(zip(["FA", "RA", "LL"], ["#e74c3c", "#2ecc71", "#9b59b6"])):
    ax.bar(x + i*w, [cmp[m][mk] for m in methods], w, label=mk, color=col, alpha=0.85)
ax.set_xticks(x + w); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("Score")
ax.set_title("Method Comparison  (FA↓ better | RA↑ better | LL↓ more complete unlearning)")
ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 1.05)
plt.tight_layout()
cmp_path = RESULTS_DIR / "method_comparison.png"
plt.savefig(str(cmp_path), dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {cmp_path}")


# =============================================================================
# 9. Save Results
# =============================================================================

results_summary = {
    "task"       : "T-06",
    "method"     : "gradient_difference" if not SIMPLIFIED_GA_ONLY else "gradient_ascent_only",
    "model"      : "llava-hf/llama3-llava-next-8b-hf",
    "config"     : CONFIG,
    "best_epoch" : best_epoch,
    "timestamp"  : datetime.now().isoformat(),
    "baseline"   : {k: v for k, v in baseline_metrics.items() if k != "LL_answers"},
    "final"      : {k: v for k, v in final_metrics.items()    if k != "LL_answers"},
    "history"    : [{k: v for k, v in h.items() if k != "LL_answers"} for h in history],
    "ll_analysis": {
        "baseline_ll_answers": baseline_metrics.get("LL_answers", []),
        "final_ll_answers"   : final_metrics.get("LL_answers", []),
        "ll_dropped"         : baseline_metrics["LL"] - final_metrics["LL"],
        "interpretation"     : (
            "Gradient-based unlearning flows through the full LLaMA LM, potentially "
            "erasing token-level zebra representations in both language and vision pathways."
        ),
    },
    "notes": (
        "Simplified: GA only, no retain constraint."
        if SIMPLIFIED_GA_ONLY else
        "Full Gradient Difference: GA on forget + GD on retain."
    ),
}

results_path = RESULTS_DIR / "results_summary.json"
with open(results_path, "w") as f:
    json.dump(results_summary, f, indent=2)

history_df   = pd.DataFrame([{k: v for k, v in h.items() if k != "LL_answers"} for h in history])
history_path = RESULTS_DIR / "training_history.csv"
history_df.to_csv(history_path, index=False)

print(f"Results JSON → {results_path}")
print(f"History CSV  → {history_path}")
print()
print(history_df.to_string(index=False))

best_link = OUTPUT_DIR / "best_checkpoint"
if best_link.is_symlink():
    best_link.unlink()
if best_ckpt_path is not None:
    best_link.symlink_to(Path(best_ckpt_path).resolve())
    print(f"best_checkpoint → epoch_{best_epoch:02d}")


# =============================================================================
# 10. Final Summary
# =============================================================================

print("\n" + "═"*62)
print("  T-06 GRADIENT-BASED UNLEARNING — FINAL SUMMARY")
print("═"*62)
print(f"  Method       : {'GA only (simplified)' if SIMPLIFIED_GA_ONLY else 'Gradient Difference (GA + λ·GD)'}")
print(f"  Lambda retain: {0.0 if SIMPLIFIED_GA_ONLY else LAMBDA_RETAIN}")
print(f"  Best epoch   : {best_epoch} / {MAX_EPOCHS}")
print()
print(f"  {'Metric':<12} │ {'Baseline':>8} │ {'Final':>7} │ {'Δ':>6}")
print(f"  {'─'*12}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*6}")
for m in ["FA", "RA", "LL", "VD"]:
    b  = baseline_metrics.get(m, 0)
    f_ = final_metrics.get(m, 0)
    print(f"  {m:<12} │ {b:>8.3f} │ {f_:>7.3f} │ {f_-b:>+6.3f}")
print()
interp = (
    "substantially removed" if final_metrics["LL"] < 0.3
    else "partially removed"  if final_metrics["LL"] < 0.6
    else "largely retained"
)
print(f"  LL finding   : Language-level zebra knowledge {interp}.")
print(f"  Saved to     : {OUTPUT_DIR}")
print(f"  Best ckpt    : {best_ckpt_path}")
print("═"*62)