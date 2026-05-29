import torch
from model import ALRConfig, ALRModel

def demonstrate_inference():
    print("--- ALR-DKR Inference Demonstration ---")
    config = ALRConfig(
        vocab_size=1000, 
        hidden_size=256, 
        num_hidden_layers=4, 
        num_attention_heads=4, 
        intermediate_size=512,
        max_recurrence_steps=10,
        recurrence_threshold=0.5
    )
    
    # Load model (untrained for prototype)
    model = ALRModel(config)
    model.eval()
    
    # We will simulate the "Internal Monologue Gate" by overriding its output for the demonstration.
    # In a real trained model, the gate naturally outputs higher probabilities for complex prompts.
    
    def run_prompt(prompt_type, simulated_gate_prob_sequence):
        print(f"\nProcessing '{prompt_type}' prompt...")
        
        # Override the gate temporarily to simulate threshold crossing over time
        # We'll use an iterator to return decreasing probabilities to simulate "thinking concluding"
        prob_iter = iter(simulated_gate_prob_sequence)
        
        original_forward = model.recurrence_gate.forward
        def mock_gate_forward(x):
            try:
                val = next(prob_iter)
            except StopIteration:
                val = 0.0 # Stop condition
            return torch.tensor([[val]], dtype=torch.float32)
            
        model.recurrence_gate.forward = mock_gate_forward
        
        # Mock input
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        
        with torch.no_grad():
             logits, recurrence_counts, history = model(input_ids)
             
        loops = int(recurrence_counts.item())
        print(f"-> Internal 'Thinking' Loops Executed: {loops} / {config.max_recurrence_steps}")
        if loops > 1:
            print("-> System detected high complexity. Deep reasoning recurrent loop engaged.")
        else:
            print("-> System detected low complexity. Shallow processing pathway engaged.")
            
        # Restore gate
        model.recurrence_gate.forward = original_forward
        
    # Simulate a simple prompt: Gate immediately outputs probability < 0.5
    run_prompt("Simple Factual Retrieval (e.g. 'What is the capital of France?')", [0.2])
    
    # Simulate a complex prompt: Gate outputs probability > 0.5 for several steps, then < 0.5 when done
    run_prompt("Complex Logical Reasoning (e.g. 'Solve this algebraic word problem...')", [0.95, 0.85, 0.70, 0.60, 0.40])
    
    # Simulate an extremely hard prompt: Hits max steps limit
    run_prompt("Extreme Edge-Case Logic (e.g. 'Write a novel structure...')", [0.99] * 15)

if __name__ == "__main__":
    demonstrate_inference()
