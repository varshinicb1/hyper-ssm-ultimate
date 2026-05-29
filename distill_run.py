import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os

# Ensure module visibility
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hyper_ssm.model import HyperSSM, HyperSSMConfig
from hyper_ssm.tokenizer import HyperTokenizer
from transformers import GPT2Config, GPT2LMHeadModel

class SyntheticEnglishDataLoader:
    """ Proxy loader to bypass slow HuggingFace internet downloads for quick tests """
    def __init__(self, tokenizer, batch_size=4, seq_len=128):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.text = "Once upon a time. " * 3000
        self.tokens = self.tokenizer.encode(self.text)['input_ids'][0].tolist()
        
    def stream_batches(self, max_batches=200):
        buffer = list(self.tokens)
        for _ in range(max_batches):
            batch_X, batch_Y = [], []
            for _ in range(self.batch_size):
                if len(buffer) < self.seq_len + 1:
                    buffer = list(self.tokens)
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                batch_X.append(chunk[:-1])
                batch_Y.append(chunk[1:])
                
            yield torch.tensor(batch_X, dtype=torch.long), torch.tensor(batch_Y, dtype=torch.long)

def distillation_loss(student_logits, teacher_logits, targets, T=2.0, alpha=0.5):
    """
    Computes Distillation Loss: combination of standard Cross Entropy and Kullback-Leibler Divergence.
    T: Temperature to soften probabilities.
    alpha: Weight for the KL Divergence term (1-alpha will be used for Cross Entropy).
    """
    # 1. Standard Cross Entropy Loss against true labels
    ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))
    
    # 2. KL Divergence Loss against Teacher Soft Targets
    # KLDivLoss expects input in log-space and target in linear space
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    
    # Use batchmean reduction for proper averaging across the batch
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    
    # Scale KL loss by T^2 to match magnitude of CE loss
    kl_loss = kl_loss * (T ** 2)
    
    return (1.0 - alpha) * ce_loss + alpha * kl_loss, ce_loss, kl_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=== HYPER-SSM KNOWLEDGE DISTILLATION ===")
    print(f"Executing on: {device}")
    
    tokenizer = HyperTokenizer()
    
    # Teacher Model: Uninitialized Proxy GPT-2
    print("Loading Proxy Teacher Model (GPT-2 structure, random weights to bypass 500MB download)...")
    teacher_config = GPT2Config(vocab_size=tokenizer.vocab_size)
    teacher_model = GPT2LMHeadModel(teacher_config).to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
        
    # Student Model: Hyper-SSM (40M parameters)
    print("Initializing Student Model (Hyper-SSM)...")
    config = HyperSSMConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_layers=12
    )
    student_model = HyperSSM(config).to(device)
    
    total_params = student_model.get_num_params()
    print(f"Student Parameters: {total_params / 1e6:.2f} M\n")
    
    # Optimizer
    # Smaller learning rate applied to protect Hyperbolic boundary mappings
    optimizer = optim.AdamW(student_model.parameters(), lr=1e-4) 
    
    # DataLoader
    batch_size = 4
    seq_len = 128
    loader = SyntheticEnglishDataLoader(tokenizer, batch_size=batch_size, seq_len=seq_len)
    
    # Ensure memory doesn't explode if we run out of VRAM (we'll just reduce batch size if needed)
    # AMP is fatal to Hyperbolic math, so we execute in full float32.
    epochs = 1
    print_interval = 20
    
    student_model.train()
    
    for epoch in range(epochs):
        print(f"\n--- Distillation Epoch {epoch+1}/{epochs} ---")
        
        for batch_idx, (X, Y) in enumerate(loader.stream_batches(max_batches=500)):
            X, Y = X.to(device), Y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass Teacher
            with torch.no_grad():
                teacher_outputs = teacher_model(X)
                teacher_logits = teacher_outputs.logits
            
            # Forward pass Student (Float32 required for hyperbolic stability)
            student_logits = student_model(X)
            
            # Compute combined Distillation Loss
            loss, ce, kl = distillation_loss(student_logits, teacher_logits, Y, T=2.0, alpha=0.5)
            
            # Backward and optimize with scaler
            loss.backward()
            
            # Gradient clipping (unscale first)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=0.01)
            
            optimizer.step()
            
            if batch_idx % print_interval == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx:05d} | Total Loss: {loss.item():.4f} | CE: {ce.item():.4f} | KL: {kl.item():.4f}")
                
        # Save distilled checkpoint
        checkpoint_path = f"hyper_ssm_distilled_ep{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Saved Distilled Checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()
