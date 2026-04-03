import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPVisionModel, CLIPImageProcessor
from datasets import load_dataset
from tqdm import tqdm
import math

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Environment
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    # LLaVA 1.5 Vision Encoder specs
    VISION_MODEL_ID = "openai/clip-vit-large-patch14-336"
    D_MODEL = 1024
    
    # SAE Architecture (TopK SAE)
    EXPANSION_FACTOR = 32
    D_SAE = D_MODEL * EXPANSION_FACTOR  # 32,768
    TOP_K = 32                          # Force sparsity to exactly 32 active features/token

    # Activation Extraction
    EXTRACT_LAYER = -2                  # LLaVA 1.5 uses the penultimate layer
    MAX_IMAGES = 50_000                 # Yields ~28.8 Million activation vectors (576 patches * 50k)
    CHUNK_SIZE = 1_000_000              # Save 1M vectors per file (~2GB per file at fp16)
    ACT_DIR = "./activations"           # Directory to store intermediate tensors

    # SAE Training Checkpointing
    BATCH_SIZE = 4096
    LEARNING_RATE = 3e-4
    EPOCHS = 1
    CKPT_DIR = "./sae_checkpoints"

# Initialize dirs
os.makedirs(Config.ACT_DIR, exist_ok=True)
os.makedirs(Config.CKPT_DIR, exist_ok=True)


# ==========================================
# PHASE 1: ACTIVATION EXTRACTION
# ==========================================
def extract_activations():
    """
    Passes images through the CLIP vision encoder and saves patch 
    activations to disk. We separate this from training to save VRAM.
    """
    print(f"\n--- Starting Phase 1: Activation Extraction ---")
    
    # We load ONLY the vision encoder used by LLaVA 1.5 (no Vicuna 7B LLM here -> saves 14GB VRAM)
    processor = CLIPImageProcessor.from_pretrained(Config.VISION_MODEL_ID)
    model = CLIPVisionModel.from_pretrained(Config.VISION_MODEL_ID, torch_dtype=Config.DTYPE).to(Config.DEVICE)
    model.eval()

    # Stream ImageNet validation (no huge disk downloads)
    dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    
    buffer = []
    chunk_idx = 0
    img_count = 0

    print("Extracting layer", Config.EXTRACT_LAYER, "patch tokens...")
    
    zebra_count = 0
    other_count = 0
    MAX_ZEBRAS = 1000
    MAX_OTHER = Config.MAX_IMAGES - MAX_ZEBRAS

    with torch.no_grad():
        for sample in tqdm(dataset):
            if zebra_count >= MAX_ZEBRAS and other_count >= MAX_OTHER:
                break
                
            is_zebra = (sample['label'] == 340)
            
            # Skip if we already have enough of this category
            if is_zebra and zebra_count >= MAX_ZEBRAS: continue
            if not is_zebra and other_count >= MAX_OTHER: continue

            img = sample['image'].convert('RGB')
            inputs = processor(images=img, return_tensors="pt").to(Config.DEVICE, dtype=Config.DTYPE)
            
            outputs = model(**inputs, output_hidden_states=True)
            layer_acts = outputs.hidden_states[Config.EXTRACT_LAYER]
            patch_acts = layer_acts[0, 1:, :] # Shape: [576, 1024]
            
            buffer.append(patch_acts.cpu())
            
            if is_zebra: zebra_count += 1
            else: other_count += 1
            
            img_count = zebra_count + other_count
            
            # If buffer hits roughly CHUNK_SIZE, save it to disk
            if sum(b.shape[0] for b in buffer) >= Config.CHUNK_SIZE:
                chunk_tensor = torch.cat(buffer, dim=0) # Shape: [CHUNK_SIZE, 1024]
                path = os.path.join(Config.ACT_DIR, f"chunk_{chunk_idx:04d}.pt")
                torch.save(chunk_tensor, path)
                print(f"Saved {path} with {chunk_tensor.shape[0]} tokens.")
                
                buffer = []
                chunk_idx += 1

    # Save remaining buffer
    if len(buffer) > 0:
        chunk_tensor = torch.cat(buffer, dim=0)
        path = os.path.join(Config.ACT_DIR, f"chunk_{chunk_idx:04d}.pt")
        torch.save(chunk_tensor, path)
        print(f"Saved {path} with {chunk_tensor.shape[0]} tokens.")

    print(f"Extraction complete. Total images: {img_count}")


# ==========================================
# PHASE 2: TOP-K SAE ARCHITECTURE
# ==========================================
class TopKSAE(nn.Module):
    """
    TopK Sparse Autoencoder (as per recent OpenAI / Anthropic scaling laws).
    Instead of an L1 penalty, we explicitly enforce sparsity by only keeping
    the top K activations. Converges faster and has no 'dead latents' collapse.
    """
    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        # Encoder & Decoder Params
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        
        self.W_dec = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform for encoder, normalized columns for decoder
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.data = torch.nn.functional.normalize(self.W_dec.data, dim=0)

    def encode(self, x):
        # pre-activations: (x - b_dec) W_enc^T + b_enc
        x_centered = x - self.b_dec
        acts = torch.nn.functional.linear(x_centered, self.W_enc, self.b_enc)
        
        # Apply TopK mask
        topk_vals, topk_indices = torch.topk(acts, self.k, dim=-1)
        h = torch.zeros_like(acts)
        h.scatter_(-1, topk_indices, torch.relu(topk_vals))
        return h

    def decode(self, h):
        return torch.nn.functional.linear(h, self.W_dec, self.b_dec)

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        
        # Loss is simply Mean Squared Error (reconstruction loss)
        # TopK natively handles the sparsity constraint!
        loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return x_hat, h, loss

    @torch.no_grad()
    def normalize_decoder(self):
        """Must be called after optimizer.step() to keep decoder weights normalized."""
        self.W_dec.data = torch.nn.functional.normalize(self.W_dec.data, dim=0)


# ==========================================
# PHASE 3: TRAINING LOOP & CHECKPOINTING
# ==========================================
class ChunkDataset(Dataset):
    def __init__(self, chunk_path):
        self.data = torch.load(chunk_path, map_location="cpu").to(torch.float32)
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def train_sae():
    """Trains the SAE on the extracted activation chunks."""
    print(f"\n--- Starting Phase 2: SAE Training ---")
    
    chunks = sorted(glob.glob(os.path.join(Config.ACT_DIR, "*.pt")))
    if not chunks:
        raise ValueError("No activation chunks found! Run extract_activations() first.")

    model = TopKSAE(Config.D_MODEL, Config.D_SAE, Config.TOP_K).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    start_chunk = 0
    start_epoch = 0

    # Resume from checkpoint if it exists
    latest_ckpt_path = os.path.join(Config.CKPT_DIR, "sae_latest.pt")
    if os.path.exists(latest_ckpt_path):
        print(f"Resuming from checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_chunk = checkpoint['chunk_idx']
        start_epoch = checkpoint['epoch']

    print(f"Architecture: TopK SAE ({Config.D_MODEL} -> {Config.D_SAE}), k={Config.TOP_K}")
    print(f"Total chunks to process: {len(chunks)}")

    model.train()
    
    for epoch in range(start_epoch, Config.EPOCHS):
        for chunk_idx in range(start_chunk, len(chunks)):
            chunk_file = chunks[chunk_idx]
            print(f"\nEpoch {epoch+1}/{Config.EPOCHS} | Training on {os.path.basename(chunk_file)}...")
            
            dataset = ChunkDataset(chunk_file)
            dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
            
            pbar = tqdm(dataloader)
            for batch in pbar:
                batch = batch.to(Config.DEVICE)
                
                optimizer.zero_grad()
                x_hat, h, loss = model(batch)
                loss.backward()
                optimizer.step()
                
                # Critical for SAEs: unit norm decoder columns
                model.normalize_decoder()
                
                pbar.set_description(f"Loss: {loss.item():.4f}")

            # Save Checkpoint after every chunk
            torch.save({
                'epoch': epoch,
                'chunk_idx': chunk_idx + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }, latest_ckpt_path)
            
        start_chunk = 0 # reset chunk index for subsequent epochs

    print("\nTraining Complete! Saving final SAE model.")
    torch.save(model.state_dict(), os.path.join(Config.CKPT_DIR, "vision_sae_final.pt"))


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # TOGGLE THESE TRUE/FALSE BASED ON WHAT YOU WANT TO RUN
    RUN_EXTRACTION = True
    RUN_TRAINING = True

    if RUN_EXTRACTION:
        extract_activations()
    
    if RUN_TRAINING:
        train_sae()
