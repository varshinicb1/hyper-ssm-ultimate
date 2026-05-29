import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Ensure module visibility
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hyper_ssm.model import HyperSSM, HyperSSMConfig
from hyper_ssm.tokenizer import HyperTokenizer

from hyper_ssm.cuda_ops import get_lorentz_product, get_project_to_tangent, is_cuda_available

lorentz_product = get_lorentz_product()
project_to_tangent = get_project_to_tangent()

if is_cuda_available():
    print("[SYSTEM] Loaded accelerated CUDA kernels for Hyperbolic mapping (via cuda_ops).")
else:
    print("[SYSTEM] Using PyTorch fallbacks for Riemannian ops.")

class SyntheticEnglishDataLoader:
    """
    A local proxy loader to bypass slow HuggingFace downloads.
    Provides identical [batch, seq_len] tensor constraints to prove the model's memory
    footprint and learning loop execution capability on an English string.
    """
    def __init__(self, tokenizer, batch_size=4, seq_len=128):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.text = (
            "Once upon a time, there was a little girl who lived in a forest. "
            "She loved to play with the animals and sing songs. "
            "One day, she met a wise old owl who taught her the secrets of magic. "
            "She used her magic to help her friends and make the forest a better place. "
            "The end. "
        ) * 1000  # Repeat to create a large string
        self.tokens = self.tokenizer.encode(self.text)['input_ids'][0].tolist()
        
    def stream_batches(self, num_batches=200):
        buffer = list(self.tokens)
        for _ in range(num_batches):
            batch_X, batch_Y = [], []
            for _ in range(self.batch_size):
                if len(buffer) < self.seq_len + 1:
                    buffer = list(self.tokens) # reset
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                batch_X.append(chunk[:-1])
                batch_Y.append(chunk[1:])
                
            yield torch.tensor(batch_X, dtype=torch.long), torch.tensor(batch_Y, dtype=torch.long)

def save_checkpoint(model, optimizer, epoch, batch_idx, loss, filename="hyper_ssm_checkpoint.pt"):
    print(f"\n[SAVE] Saving model checkpoint to {filename}...")
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)

def generate_text(model, tokenizer, prompt="Once upon a time", max_new_tokens=50, device='cuda'):
    """
    Autoregressive generation loop to empirically verify the model is learning English syntax.
    """
    model.eval()
    print(f"\n--- Generation Test: '{prompt}' ---")
    
    # 1. Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')['input_ids'].to(device)
    
    # 2. Autoregressive Loop
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            logits = model(input_ids)
            
            # Get the logits of the very last token in the sequence
            next_token_logits = logits[:, -1, :]
            
            # Simple greedy decoding for testing (argmax)
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to running sequence
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            # If the model predicts End of Sequence (EOS), stop early
            if next_token_id.item() == tokenizer.tokenizer.eos_token_id:
                break
                
    # 3. Decode back to text
    generated_text = tokenizer.decode(input_ids[0])
    print(f"{generated_text}\n-----------------------------------\n")
    model.train()

def main():
    print("=== HYPER-SSM STARTING SCALE-UP (Natural Language) ===\n")
    
    # Select Device (GPU if available, CPU otherwise)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on: {device}")
    
    tokenizer = HyperTokenizer()
    
    # Scale-Up Configuration: ~50M Parameters
    config = HyperSSMConfig(
        vocab_size=tokenizer.vocab_size, 
        hidden_size=256,    # Scaled up mapped geometry
        num_layers=12       # Scaled up depth
    )
    
    # Initialize Model
    model = HyperSSM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params / 1e6:.2f} M\n")
    
    # Initialize Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Enable BF16 processing as required for massive hyperbolic exponentiation bounds
    print("Enabling strict bfloat16 mixed precision execution for geometric stability...")
    torch.set_float32_matmul_precision("medium")
    model = model.bfloat16()
    
    # Initialize Synthetic Proxy DataLoader
    batch_size = 4
    seq_len = 128
    loader = SyntheticEnglishDataLoader(tokenizer, batch_size=batch_size, seq_len=seq_len)
    
    epochs = 3
    print_interval = 20
    generate_interval = 200 # Generate a story every 200 batches
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        model.train()
        
        # Stream stories out of the HuggingFace datasets iterator
        batch_idx = 0
        for X, Y in loader.stream_batches():
            X, Y = X.to(device), Y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: Unpack Logits and accumulated Entropy Regularization
            # Executed entirely in bfloat16 space
            logits, entropy = model(X, return_entropy=True) # [batch, seq_len, vocab_size]
            
            # Calculate Task Loss over the vocab dimension
            task_loss = criterion(logits.view(-1, config.vocab_size), Y.view(-1))
            
            # Formulate final loss with penalizations to prevent Router Selection Collapse
            loss = task_loss - 0.01 * entropy
            loss.backward()
            
            # Riemannian Gradient Adjustments
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # Identify parameters processing hyperbolic spatial coordinates
                        if 'W_state.weight' in name or 'W_input.weight' in name:
                            # Project Euclidean gradients onto the Riemannian manifold tangent space
                            # to prevent the optimizer from dragging points off the hyperboloid sheet
                            param.grad = project_to_tangent(param, param.grad)
                            
            # Euclidean clipping of tangent-space projected bounds
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            optimizer.step()
            
            # Print Loss
            if batch_idx % print_interval == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx:05d} | Loss: {loss.item():.4f} | R-Entropy: {entropy.item():.4f}")
                
            # Run empirical Generation Test
            if batch_idx > 0 and batch_idx % generate_interval == 0:
                generate_text(model, tokenizer, prompt="Once upon a time, there was a little", device=device)
                
            batch_idx += 1
            
        # Save a checkpoint at the end of every epoch
        save_checkpoint(model, optimizer, epoch, batch_idx, loss.item(), filename=f"hyper_ssm_epoch_{epoch+1}.pt")
            
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
