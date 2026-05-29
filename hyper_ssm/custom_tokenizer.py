import json
import os
from collections import defaultdict

class CustomBPETokenizer:
    """
    A pure-Python implementation of Byte-Pair Encoding (BPE).
    Completely severs the dependency on external architectures (like GPT-2)
    allowing the Hyper-SSM to learn a vocabulary strictly optimized for its target dataset.
    
    Operates initially on raw UTF-8 byte representations, sequentially merging the most frequent 
    byte pairs into new tokens until the target `vocab_size` is reached.
    """
    def __init__(self):
        # The base vocabulary is the 256 strict UTF-8 byte values (0 to 255)
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {} # (int, int) -> int (new token id)
        self.special_tokens = {
            "<|endoftext|>": 256,
            "<|pad|>": 257
        }
        
        # Add special tokens to the core vocab dictionary to prevent crashing during decoding
        for token_text, token_id in self.special_tokens.items():
            self.vocab[token_id] = token_text.encode("utf-8")
            
        self.eos_token_id = self.special_tokens["<|endoftext|>"]
        self.pad_token_id = self.special_tokens["<|pad|>"]
        
        self.next_token_id = 258

    def _get_stats(self, ids):
        """
        Count the frequency of all adjacent pairs of tokens in the given sequence.
        Returns a dictionary mapping (token1, token2) -> frequency
        """
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def _merge(self, ids, pair, idx):
        """
        Replace all consecutive occurrences of `pair` with the new token `idx` 
        in the sequence `ids`.
        Returns the compressed sequence.
        """
        new_ids = []
        i = 0
        while i < len(ids):
            # If we are not at the very last token, check if the current and next match the target pair
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2 # Skip the matched pair
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, target_vocab_size, verbose=False):
        """
        Trains the BPE vocabulary on the provided text string.
        Will compress the UTF-8 byte streams iteratively until `target_vocab_size` is reached.
        """
        assert target_vocab_size >= 258, "Target vocabulary size must be greater than strict UTF-8 base (258)"
        num_merges = target_vocab_size - 258
        
        # 1. Convert initial text into raw UTF-8 byte values (0-255)
        # This gives us our starting mathematical state.
        raw_bytes = text.encode("utf-8")
        ids = list(raw_bytes)
        
        if verbose:
            print(f"[BPE Train] Input text length: {len(text)} characters.")
            print(f"[BPE Train] UTF-8 Byte stream length: {len(ids)} bytes.")
            print(f"[BPE Train] Attempting to learn {num_merges} new tokens...")

        # 2. Iteratively merge the most frequently occurring byte pairs
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                if verbose: print("[BPE Train] No more pairs to merge. Exiting early.")
                break
                
            # Find the exact pair with the highest frequency count
            top_pair = max(stats, key=stats.get)
            
            # Map this pair to our next available sequential token ID
            new_id = self.next_token_id + i
            
            # Record the merge operation 
            self.merges[top_pair] = new_id
            
            # Record the exact byte sequence this new token represents
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            
            # Overwrite the sequence in memory with the newly compressed token
            ids = self._merge(ids, top_pair, new_id)
            
            if verbose and i % max(1, (num_merges//10)) == 0:
                print(f"Merge {i+1}/{num_merges}: {top_pair} -> {new_id} ({stats[top_pair]} occurrences) | New Seq Len: {len(ids)}")
                
        # Update the mathematical tracker
        self.next_token_id += num_merges

    def encode(self, text):
        """
        Translates a string into a compressed token sequence using the learned `merges`.
        """
        # 1. Base conversion
        raw_bytes = text.encode("utf-8")
        ids = list(raw_bytes)
        
        # 2. Iterative compression
        # We must apply the merges in the exact same order they were learned during training 
        # (which is mathematically guaranteed by iterating over our ordered dictionary logic, assuming Python 3.7+).
        # We keep iterating until no more valid pairwise merges can be completed.
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            
            # We want to find the pair in `stats` that was merged *earliest* during training.
            # To do this safely, we ask our `self.merges` dictionary which known pair has the lowest target ID.
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break # None of the current sequence pairs exist in our learned dictionary
                
            # Perform the mathematical compression of the sequence
            new_id = self.merges[pair]
            ids = self._merge(ids, pair, new_id)
            
        return ids

    def decode(self, ids):
        """
        Translates a sequence of token IDs back into a human-readable UTF-8 string.
        """
        raw_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = raw_bytes.decode("utf-8", errors="replace") # Replace invalid byte alignments gracefully
        return text

    def save(self, prefix="custom"):
        """ Saves the learned vocabulary and merges to disk for persistent caching. """
        os.makedirs("tokenizer_cache", exist_ok=True)
        # We save merges as "idx1 idx2 -> resulting_idx" lines to avoid JSON tuple serialization issues
        with open(f"tokenizer_cache/{prefix}_merges.txt", "w", encoding="utf-8") as f:
            for (p0, p1), idx in self.merges.items():
                f.write(f"{p0} {p1} {idx}\n")
        
        # Special tokens dictionary
        with open(f"tokenizer_cache/{prefix}_special.json", "w", encoding="utf-8") as f:
            json.dump(self.special_tokens, f)

    def load(self, prefix="custom"):
        """ Loads a previously trained Tokenizer from disk. """
        merge_file = f"tokenizer_cache/{prefix}_merges.txt"
        special_file = f"tokenizer_cache/{prefix}_special.json"
        
        if not os.path.exists(merge_file) or not os.path.exists(special_file):
            return False
            
        # 1. Rebuild base vocabulary
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        
        # 2. Reload special tokens
        with open(special_file, "r", encoding="utf-8") as f:
            self.special_tokens = json.load(f)
            
        for text, idx in self.special_tokens.items():
            self.vocab[idx] = text.encode("utf-8")
            
        self.eos_token_id = self.special_tokens["<|endoftext|>"]
        self.pad_token_id = self.special_tokens["<|pad|>"]
        
        # 3. Reload merges and mathematically reconstruct the compound string vocabulary
        highest_id = 257
        with open(merge_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    p0, p1, idx = int(parts[0]), int(parts[1]), int(parts[2])
                    self.merges[(p0, p1)] = idx
                    self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
                    highest_id = max(highest_id, idx)
                    
        self.next_token_id = highest_id + 1
        return True
