import torch
import torch.optim as optim
from model import ALRConfig, ALRModel
from loss import OrthogonalKnowledgeLoss

def train_mock_step():
    print("Setting up training environment...")
    
    # Tiny Configuration for fast prototyping test
    config = ALRConfig(
        vocab_size=1000, 
        hidden_size=256, 
        num_hidden_layers=2, 
        num_attention_heads=4, 
        intermediate_size=512,
        max_recurrence_steps=3 # Allow up to 3 latent thinking loops
    )
    
    model = ALRModel(config)
    
    criterion = OrthogonalKnowledgeLoss(penalty_weight=0.001)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Mock Data: Batch Size 4, Sequence Length 16
    batch_size = 4
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Target is just shifted input_ids
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    
    print("\n--- Starting Mock Training Step ---")
    
    optimizer.zero_grad()
    
    # Forward Pass
    logits, recurrence_counts, history = model(input_ids)
    
    # Loss Calculation
    total_loss, ce_loss, reg_loss = criterion(logits, target_ids, list(model.parameters()))
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Cross Entropy Loss: {ce_loss:.4f} | Orthogonal Penalty: {reg_loss:.4f}")
    print(f"Average Internal Recurrence Loops Used: {recurrence_counts.mean().item():.2f} / {config.max_recurrence_steps}")
    
    # Backward Pass
    print("Running Backward Pass to verify graph integrity...")
    total_loss.backward()
    
    # Update weights
    optimizer.step()
    print("Optimizer step successful. Weights updated.")
    
    print("--- Training Step Complete! Architecture is valid! ---")

if __name__ == "__main__":
    train_mock_step()
