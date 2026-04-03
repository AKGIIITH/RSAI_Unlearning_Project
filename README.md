# RSAI Unlearning Project

A comprehensive machine unlearning framework for Vision-Language Models (VLMs), specifically targeting the LLaVA model to remove zebra knowledge while preserving general capabilities.

## Project Overview

This project implements and compares multiple unlearning strategies for VLMs, ranging from feature-level interventions to gradient-based methods. The framework systematically evaluates each approach across four key metrics:

- **FA** (Forget Accuracy): % of forget-set answers still mentioning the target (lower = better unlearning)
- **RA** (Retain Accuracy): % of retain-set answers that remain correct (higher = better)
- **LL** (Language Leakage): % of text-only probes leaking zebra knowledge (lower = better removal)
- **AR** (Adversarial Robustness): % of adversarial probes still identifying the target (lower = better)

---

## Tasks Overview

### **Task 0: Sparse Autoencoder (SAE) Training**
**Goal**: Train SAEs on CLIP vision encoder activations to decompose them into interpretable features.

- Extracts patch activations from LLaVA's vision tower (Layer -2)
- Implements multiple SAE architectures:
  - ReLU-based SAE with L1 sparsity
  - TopK SAE (fixed sparsity with k=32)
- Trains on ~12 million activation vectors from ImageNet + COCO
- Saves weights to HuggingFace Hub for downstream tasks

**Key Files**: `task_0.py`, `task_0.ipynb`

---

### **Task 1: SAE Testing & Model Verification**
**Goal**: Load and validate CLIP models + SAEs for downstream analysis.

- Loads pre-trained CLIP (ViT-B-32)
- Sets up forward hooks to extract intermediate activations
- Analyzes activation statistics to inform SAE architecture decisions
- Verifies image-text similarity scoring

**Key Files**: `task_1.ipynb`

---

### **Task 2: Baseline Evaluation (No Unlearning)**
**Goal**: Establish baseline metrics on the original, unmodified LLaVA model.

- Constructs dataset:
  - **Forget set**: 80 zebra images
  - **Retain set**: 100 horse/donkey images
  - **Adversarial probes**: 50 zebra images with blur/greyscale treatments
  - **Text-only probes**: 25 zebra-knowledge questions
  
- Evaluates model on all four metrics
- Saves responses to `task_2_results/baseline_responses.json` for comparison

**Key Files**: `task_2.ipynb`

**Baseline Metrics**:
- **FA**: 98.67% (model still identifies zebras)
- **RA**: 71.33% (general knowledge intact)
- **LL**: 96.00% (LLM retains textual zebra knowledge)
- **AR**: 50.00% (adversarial robustness)

---

### **Task 3: Zero Ablation on Vision Model**
**Goal**: Remove zebra-specific features from the vision encoder by zeroing them out.

**Method**:
1. Identify top-N zebra-selective SAE features via contrastive activation analysis
2. Install a forward hook on `vision_tower.vision_model.encoder.layers[-1]`
3. For each forward pass:
   - Encode hidden states through SAE
   - Zero out target zebra features (set to 0)
   - Decode and return modified activations

**Expected Results**: Lower FA (vision-level unlearning) with preserved RA and unchanged LL.

**Key Files**: `task_3.ipynb`

---

### **Task 4: Negative Steering on Vision Model**
**Goal**: Apply fine-grained steering to zebra features using SAE decoder directions.

**Method**:
1. Discover zebra-selective features via contrastive learning (high activation on zebra, low on horses/donkeys)
2. Compute baseline activation means on retain set
3. Apply negative clamping during forward pass:
   - Encode hidden states through SAE
   - Clamp target features to: `-scale × baseline_mean`
   - Use **error-term approach**: only add the reconstruction difference to hidden states
4. Multi-GPU distribution (2× T4 GPUs) for efficiency

**Expected Results**: Softer unlearning than zero ablation, better RA preservation.

**Key Files**: `task_4-v3.ipynb`

---

### **Task 5: Language Model Steering**
**Goal**: Intervene at the LLM layer (LLaVA Layer 24) to suppress zebra representations.

**Method**:
1. Load pre-trained SAE at LLaVA LM Layer 24
2. Use symmetrical minimal pairs ("This is a photo of a zebra" vs. "horse") to identify zebra-specific LM features
3. Apply negative clamping hook on LM hidden states
4. Multi-GPU distribution (2× T4 GPUs) for efficiency

**Results** (from `task_5.ipynb`):
- **FA**: 0.00% ✓ (vision features intact, LM doesn't affect vision output)
- **RA**: 50.00% (some language knowledge removed)
- **LL**: 60.00% ↓ (text-level unlearning achieved)
- **AR**: 0.00% ✓ (adversarial robustness preserved)

**Key Files**: `task_5.ipynb`

---

### **Task 6: Gradient-Based Unlearning (Both Modalities)**
**Goal**: Apply gradient-based optimization to unlearn zebra knowledge across all model parameters.

**Method**:
1. Gradient Ascent (GA) on forget set: maximize loss to force unlearning
2. Gradient Descent (GD) on retain set: minimize loss to preserve knowledge
3. Total loss: `-loss_forget + λ × loss_retain` (λ = 1.0)

**Architecture**:
- LoRA adapters on vision tower attention projections
- Language model frozen (no direct modification)
- Multimodal projector trainable

**Hyperparameters**:
- Batch size: 1 (memory constraints)
- Learning rate: 5e-4
- Max epochs: 1
- Gradient accumulation: 8 steps
- Early stopping: FA target < 0.30, RA drop > 15%

**Key Files**: `task_6.py`

---

## Directory Structure

```
RSAI_Unlearning_Project/
├── task_0.py                          # SAE training (production code)
├── task_0.ipynb                       # SAE training (notebook version)
├── task_1.ipynb                       # SAE testing & verification
├── task_2.ipynb                       # Baseline evaluation
├── task_3.ipynb                       # Zero ablation (vision)
├── task_4-v3.ipynb                    # Negative steering (vision)
├── task_5.ipynb                       # LM steering
├── task_6.py                          # Gradient-based unlearning
├── data/                              # Dataset (forget/retain images + manifest)
│   ├── forget/                        # 80 zebra images
│   ├── retain/                        # 100 horse/donkey images
│   └── dataset_manifest.json          # VQA structure & labels
├── probe_images/                      # Adversarial & text probes
│   ├── original/                      # Original zebra images
│   ├── blur/                          # Gaussian blur treatment
│   └── grey/                          # Greyscale treatment
├── features-identified-sae/           # SAE feature analysis results
│   ├── concept_features_fixed.json    # Top-N zebra features
│   └── baseline_stats_fixed.json      # Feature activation statistics
├── task_*_results/                    # Results from each task
│   ├── baseline_responses.json        # Task 2 baseline metrics
│   ├── sae_zero_ablation_results.json # Task 3 results
│   └── ...
├── Phase_2_presentation/              # Mid-project presentation (PPT)
├── Phase_3_presentation/              # Final presentation (PPT)
├── vision_sae_final.pt                # Trained SAE weights
└── README.md                          # This file
```

---

## Key Concepts

### **Sparse Autoencoders (SAEs)**
Learn to decompose dense activations into sparse, interpretable features. Useful for:
- Identifying which model features encode specific concepts (zebra stripes, etc.)
- Targeted interventions at the feature level
- Understanding model internals

**Formula**:
```
Encoder: h = ReLU(W_enc @ (x - b_dec) + b_enc)   [sparse]
Decoder: x̂ = W_dec @ h + b_dec                   [reconstruction]
Loss: MSE(x̂, x) + λ·L1(h)
```

**TopK variant** (used in Task 0):
```
Encoder: h = top_k_mask(pre_activations)         [exactly k non-zero]
Decoder: x̂ = W_dec @ h + b_dec
Loss: MSE(x̂, x)  [no L1 penalty needed]
```

### **Zero Ablation**
Set identified feature activations to zero → remove feature from model computation.
- **Pros**: Interpretable, direct, immediate effect
- **Cons**: Hard cutoff may cause artifacts or mode collapse

### **Negative Steering**
Clamp feature activations to negative values (opposite direction in feature space).
- **Pros**: Smoother transitions, preserves model coherence
- **Cons**: Less direct than ablation, may be weaker

### **Gradient-Based Unlearning**
Optimize model parameters directly using task-specific gradients.
- **Pros**: End-to-end, full model adaptation, theoretically principled
- **Cons**: High computational cost, significant collateral damage risk, harder to diagnose

---

## Running the Pipeline

### **Step 1: Train SAE** (Optional; pre-trained weights available)
```bash
python task_0.py
```
Trains on 10k COCO + 1k concept images. Outputs saved to HuggingFace Hub.

### **Step 2: Evaluate Baseline** (No unlearning)
```bash
jupyter notebook task_2.ipynb
```
Establishes the four baseline metrics (FA/RA/LL/AR).

### **Step 3: Run Unlearning Methods** (Compare approaches)
```bash
# Vision-only zero ablation
jupyter notebook task_3.ipynb

# Vision-only negative steering
jupyter notebook task_4-v3.ipynb

# Language model steering
jupyter notebook task_5.ipynb

# Full gradient-based unlearning
python task_6.py
```

Each task outputs detailed results to `task_*_results/` directory.

---

## Results Summary

| Method | FA ↓ | RA ↑ | LL ↓ | AR ↓ | Modality | Status |
|--------|------|------|------|------|----------|--------|
| Baseline | 98.67% | 71.33% | 96.00% | 50.00% | N/A | ✓ Reference |
| Zero Ablation | TBD | TBD | ≈96% | TBD | Vision only | ✓ Implemented |
| Negative Steering | TBD | TBD | ≈96% | TBD | Vision only | ✓ Implemented |
| LM Steering | 0.00% | 50.00% | 60.00% | 0.00% | LM only | ✓ Complete |
| Gradient-Based | TBD | TBD | TBD | TBD | Both | ✓ Implemented |

---

## Dependencies

```
torch >= 2.0
transformers >= 4.36
datasets
pillow
bitsandbytes >= 0.43.3
accelerate
peft
huggingface_hub
```

### Install:
```bash
pip install -q torch transformers datasets pillow bitsandbytes accelerate peft huggingface_hub
```

---

## Hardware Requirements

- **Recommended**: 2× NVIDIA T4 (16GB each) or 1× A100 (40GB)
- **Tested on**: Kaggle notebooks with 2× T4 + TPU fallback
- **Memory optimization**: 4-bit quantization, gradient checkpointing, lazy data loading

### Typical VRAM Usage:
- Model loading: ~14GB
- SAE + inference: ~2GB
- Training batch: ~4GB
- Total headroom: ~2GB (for safety)

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `task_0.py` | SAE training (production) |
| `task_6.py` | Gradient-based unlearning |
| `features-identified-sae/concept_features_fixed.json` | Top zebra SAE features |
| `data/dataset_manifest.json` | Dataset metadata & VQA structure |
| `task_2_results/baseline_responses.json` | Baseline evaluation results |

---

## Notes & Tips

- **Reproducibility**: All tasks use `seed=42` for deterministic results
- **SAE Weights**: Pre-trained available on HuggingFace Hub
- **Memory Issues**: If OOM occurs, reduce batch size in the respective notebook
- **GPU Allocation**: Use `device_map="auto"` for multi-GPU distribution
- **Long Running**: Tasks 3-6 may take 30-60 minutes on T4; use Kaggle for free GPU time
- **Metric Tracking**: All results saved to JSON for easy comparison across methods

---

## Project Workflow

```
┌─────────────────┐
│   Task 0: SAE   │  Extract activations → Train SAE
│    Training     │  on 12M vectors
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Task 1: SAE   │  Load models & verify
│   Testing       │  activation shapes
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Task 2:       │  Establish baseline
│   Baseline      │  metrics (FA/RA/LL/AR)
└────────┬────────┘
         │
    ┌────┴─────┬──────────┬─────────┐
    │           │          │         │
    ▼           ▼          ▼         ▼
┌───────┐  ┌────────┐  ┌────────┐  ┌───────┐
│Task 3:│  │Task 4: │  │Task 5: │  │Task 6:│
│ Zero  │  │Negative│  │  LM    │  │Gradient
│Ablation│  │Steering│  │Steering│  │-Based│
└───────┘  └────────┘  └────────┘  └───────┘
    │           │          │         │
    └───────────┴──────────┴─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Compare Results  │
    │ & Conclusions    │
    └──────────────────┘
```

---

## References

- **Sparse Autoencoders**: ["Scaling and evaluating sparse autoencoders"](https://arxiv.org/abs/2406.04093)
- **SAEs in Vision**: SAEs_in_vision_models.pdf
- **Steering in CLIP**: Steering_in_CLIP.pdf
- **Martyoshka SAEs**: Martyoshka_SAE.pdf
- **Machine Unlearning**: SISA/Fisher information-based approaches

---

## License & Attribution

Research project for RSAI (Representation and Semantic AI Lab). All code implements published methods in sparse autoencoders and machine unlearning.

---

**Last Updated**: April 2026  
**Project Status**: Active (Tasks 1-5 complete,