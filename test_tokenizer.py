import os
import sys

# Ensure module visibility
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hyper_ssm.custom_tokenizer import CustomBPETokenizer

def main():
    print("=== Testing Custom BPE Tokenizer ===\n")
    
    # 1. Initialize
    tokenizer = CustomBPETokenizer()
    
    # 2. Train on a small corpus
    text = (
        "Hello world! This is a test of the Hyper-SSM Custom BPE Tokenizer. "
        "Hello world again! Tokenizing is fun. "
        "The Hyper-SSM architecture scales massively."
    ) * 10 
    
    print("Training BPE vocabulary... (Target: 300 tokens)")
    tokenizer.train(text, target_vocab_size=300, verbose=True)
    
    print(f"\nTraining Complete. Total Vocab Size: {len(tokenizer.vocab)}")
    
    # 3. Test Encoding
    test_string = "Hello world! The Tokenizer scales."
    print(f"\nOriginal String: '{test_string}'")
    
    encoded = tokenizer.encode(test_string)
    print(f"Encoded IDs: {encoded}")
    print(f"Encoded Length: {len(encoded)} (Original Byte Length: {len(test_string.encode('utf-8'))})")
    
    # 4. Test Decoding
    decoded = tokenizer.decode(encoded)
    print(f"Decoded String: '{decoded}'")
    
    assert test_string == decoded, "ERROR: Lossless decoding failed!"
    print("\n[SUCCESS] Lossless Encoding -> Decoding chain verified.")
    
    # 5. Test Save / Load
    print("\nTesting I/O Caching...")
    tokenizer.save(prefix="test")
    
    new_tokenizer = CustomBPETokenizer()
    success = new_tokenizer.load(prefix="test")
    assert success, "ERROR: Failed to load tokenizer from cache."
    
    new_encoded = new_tokenizer.encode(test_string)
    assert new_encoded == encoded, "ERROR: Loaded tokenizer produced different IDs!"
    print("[SUCCESS] Save/Load caching verified.")

if __name__ == "__main__":
    main()
