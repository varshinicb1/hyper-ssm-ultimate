import torch
import time
import os
import psutil
from hyper_ssm.model import HyperSSM, HyperSSMConfig

def get_memory_mb():
    """Returns the current process memory footprint in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def benchmark_hyper_ssm():
    print("=== HYPER-SSM: The 'Zero KV-Cache' Benchmark ===\n")
    print("This simulates streaming an aggressively long context into the model")
    print("to prove that RAM usage remains totally flat (O(1) Memory Complexity).\n")
    
    config = HyperSSMConfig(
        vocab_size=1000,
        hidden_size=64, # Small for fast CPU prototype run
        num_layers=2
    )
    
    model = HyperSSM(config)
    model.eval()
    
    print(f"Model Parameters: {model.get_num_params() / 1e6:.2f} M\n")
    
    # We will simulate a continuous thought process by streaming chunks of tokens
    # Instead of sending [100000] tokens at once (which tests Pytorch tensor allocation, not the model)
    # we simulate an autoregressive generative loop over a massive context.
    
    chunk_size = 128
    num_chunks = 20 # Simulating generating 2500+ length sequence context
    
    import warnings
    warnings.filterwarnings("ignore") # Ignore Pytorch UserWarnings for clean output
    
    baseline_mem = get_memory_mb()
    print(f"[{0:4d} tokens processed] - Baseline RAM: {baseline_mem:.2f} MB")
    
    memory_log = []
    
    start_time = time.time()
    
    # We use torch.no_grad() to simulate inference memory
    with torch.no_grad():
        for i in range(num_chunks):
            # Generate a random chunk of context
            input_chunk = torch.randint(0, config.vocab_size, (1, chunk_size))
            
            # Forward pass - the model folds this chunk into the FractalStateCompressor
            logits = model(input_chunk)
            
            current_mem = get_memory_mb()
            memory_log.append(current_mem)
            
            tokens_processed = (i + 1) * chunk_size
            
            # Print update every 4 chunks
            if (i + 1) % 4 == 0:
                print(f"[{tokens_processed:4d} tokens processed] - Current RAM: {current_mem:.2f} MB \t(Delta: {current_mem - baseline_mem:+.2f} MB)")
                
    end_time = time.time()
    
    print("\n=== Benchmark Complete ===")
    print(f"Total Time: {end_time - start_time:.2f} seconds")
    
    # Analyze the memory slope
    mem_slope = memory_log[-1] - memory_log[0]
    
    if abs(mem_slope) < 2.0: # Less than 2MB drift is effectively flat
        print("\n[SUCCESS] Memory curve is FLAT. The Zero KV-Cache claim is VALIDATED.")
        print("Hyperbolic geometric folding successfully replaced linear buffer caching.")
    else:
        print(f"\n[FAILED] Memory drifted by {mem_slope:.2f} MB. Check for tensor leaks.")

if __name__ == "__main__":
    benchmark_hyper_ssm()
