import torch
import os
from transformers import GPT2Tokenizer
from .custom_tokenizer import CustomBPETokenizer

class HyperTokenizer:
    """
    A unified wrapper for the Hyper-SSM.
    If a custom BPE vocabulary is trained and cached locally, it utilizes the native 
    CustomBPETokenizer. Otherwise, it defaults to the HuggingFace GPT-2 50,257 tokenizer.
    """
    def __init__(self, use_custom=True):
        self.custom_cache_path = "tokenizer_cache/custom_merges.txt"
        
        if use_custom and os.path.exists(self.custom_cache_path):
            print("[Tokenizer] Loading native CustomBPETokenizer from local cache...")
            self.is_custom = True
            self.tokenizer = CustomBPETokenizer()
            self.tokenizer.load(prefix="custom")
            self.vocab_size = self.tokenizer.next_token_id
            self.pad_token_id = self.tokenizer.pad_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
        else:
            print("[Tokenizer] Custom BPE cache not found. Flowing back to GPT-2 HuggingFace Vocab...")
            self.is_custom = False
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            
            # Ensure we have a padding token for batching
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.vocab_size = len(self.tokenizer)
            self.pad_token_id = self.tokenizer.pad_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
        
    def encode(self, text, return_tensors='pt', max_length=None, padding=True):
        """
        Converts text strings into token IDs universally.
        """
        if self.is_custom:
            # Native Custom BPE Encoder
            token_list = self.tokenizer.encode(text)
            
            # Optionally truncate
            if max_length:
                token_list = token_list[:max_length]
                
            if return_tensors == 'pt':
                # Wrap list output in a batch formulation [1, seq_len] similar to HF
                return {'input_ids': torch.tensor([token_list], dtype=torch.long)}
            else:
                return {'input_ids': token_list}
        else:
            # Fallback HF GPT-2 Encoder
            return self.tokenizer(
                text, 
                return_tensors=return_tensors, 
                truncation=True if max_length else False,
                max_length=max_length,
                padding=padding
            )
            
    def decode(self, token_ids):
        """
        Converts token IDs back into readable English text universally.
        """
        if isinstance(token_ids, torch.Tensor):
            # If batch, decode first sequence
            if len(token_ids.shape) == 2:
                token_ids = token_ids[0]
                
            token_ids = token_ids.tolist()
            
        if self.is_custom:
            return self.tokenizer.decode(token_ids)
        else:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
