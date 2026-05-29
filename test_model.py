import torch
from model import ALRConfig, ALRModel

def test_alr_model():
    print("Initializing ALRConfig...")
    config = ALRConfig(
        vocab_size=1000, 
        hidden_size=256, 
        num_hidden_layers=4, 
        num_attention_heads=4, 
        intermediate_size=512,
        max_recurrence_steps=5
    )
    
    print("Initializing ALRModel...")
    model = ALRModel(config)
    
    print(f"Total Parameters: {model.get_num_params() / 1e6:.2f} M")
    
    # Mock input batch of 2 sequences, length 10
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    
    print("Running forward pass...")
    logits, recurrence_counts, history = model(input_ids)
    
    print(f"Logits shape: {logits.shape} (Expected: 2, 10, {config.vocab_size})")
    print(f"Average recurrences per sequence: {recurrence_counts.mean().item()}")
    
    assert logits.shape == (2, 10, config.vocab_size), "Output shape mismatch!"
    print("Forward pass successful.")

if __name__ == "__main__":
    test_alr_model()
