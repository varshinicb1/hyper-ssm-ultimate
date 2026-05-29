import torch
import sys
import os
import argparse

# Ensure module visibility
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hyper_ssm.model import HyperSSM, HyperSSMConfig
from hyper_ssm.tokenizer import HyperTokenizer

def main():
    parser = argparse.ArgumentParser(description="Hyper-SSM Inference Generator")
    parser.add_argument("--checkpoint", type=str, default="hyper_ssm_epoch_1.pt", help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time, in a magical forest", help="Starting prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K token sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p Nucleus sampling")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Hyper-SSM Inference on {device}...")
    
    tokenizer = HyperTokenizer()
    
    # Needs to match the 40M parameter configuration
    config = HyperSSMConfig(
        vocab_size=tokenizer.vocab_size, 
        hidden_size=256,
        num_layers=12
    )
    
    model = HyperSSM(config).to(device)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded weights from {args.checkpoint} (Epoch {checkpoint.get('epoch', 'N/A')})")
    else:
        print(f"WARNING: Checkpoint '{args.checkpoint}' not found! Generating with untrained chaotic weights.")

    print("\n" + "="*50)
    print(f"PROMPT: {args.prompt}")
    print("="*50)
    
    input_ids = tokenizer.encode(args.prompt, return_tensors='pt')['input_ids'].to(device)
    
    # Generate text using the built-in HyperSSM sampler
    output_ids = model.generate(
        input_ids, 
        max_new_tokens=args.max_tokens, 
        temperature=args.temperature, 
        top_k=args.top_k, 
        top_p=args.top_p
    )
    
    generated_text = tokenizer.decode(output_ids[0])
    print(f"\n{generated_text}")
    print("\n" + "="*50)

if __name__ == '__main__':
    # usage: python generate_text.py --prompt "The quick brown fox" --temperature 0.7
    main()
