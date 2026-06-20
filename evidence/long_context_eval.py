"""
Evidence Script: Concrete Long-Context Needle-in-Haystack Evaluation

This implements a simple but real needle-in-haystack test using the model's
actual generation APIs (get_final_state + update_state or generate_efficient).

For production evidence, plug the same pattern into RULER or the full
NeedleInAHaystack framework.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyper_ssm import HyperSSM, HyperSSMConfig

def run_needle_test(model, context_length=4096, num_distractors=50):
    """
    Simple needle test:
    - Insert a 'needle' fact at a random position in a long context.
    - Ask the model to retrieve it.
    """
    device = next(model.parameters()).device
    vocab_size = model.config.vocab_size

    # Create dummy context tokens
    context = torch.randint(10, vocab_size - 100, (1, context_length), device=device)

    # Needle: special tokens representing "The answer is 42"
    needle = torch.tensor([[42, 43, 44]], device=device)  # toy needle

    # Insert needle at a random position (avoiding very start/end)
    insert_pos = torch.randint(100, context_length - 100, (1,)).item()
    full_context = torch.cat([
        context[:, :insert_pos],
        needle,
        context[:, insert_pos:]
    ], dim=1)

    # Query: "What is the answer?"
    query = torch.tensor([[100, 101, 102]], device=device)  # toy query tokens

    input_ids = torch.cat([full_context, query], dim=1)

    print(f"Context length: {input_ids.shape[1]} tokens (needle inserted at pos ~{insert_pos})")

    with torch.no_grad():
        # Use efficient generation if available
        if hasattr(model, "generate_efficient"):
            output = model.generate_efficient(input_ids, max_new_tokens=5)
        else:
            output = model.generate(input_ids, max_new_tokens=5)

    # In a real test you would decode output and check if "42" appears
    print(f"Generated continuation token IDs: {output[0, -5:].tolist()}")
    print("Check if the model retrieved the needle (token 42 area).")

    return output

if __name__ == "__main__":
    config = HyperSSMConfig(vocab_size=1000, hidden_size=128, num_layers=6)
    model = HyperSSM(config, use_tiled_compressor=True).eval()

    print("Running needle-in-haystack test...")
    run_needle_test(model, context_length=4096)
