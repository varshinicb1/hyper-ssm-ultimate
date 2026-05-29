import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ALRConfig:
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=1024,
        num_hidden_layers=12,  # Physical layers
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=2048,
        max_recurrence_steps=10, # Max loops for the internal monologue
        recurrence_threshold=0.5, # Probability threshold to continue looping
        dropout_prob=0.1
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.max_recurrence_steps = max_recurrence_steps
        self.recurrence_threshold = recurrence_threshold
        self.dropout_prob = dropout_prob

class LatentRecurrenceGate(nn.Module):
    """
    Evaluates the entropic complexity of the hidden state and decides if the 
    latent state should be looped back through the main model blocks again.
    """
    def __init__(self, config: ALRConfig):
        super().__init__()
        # A simple lightweight classifier that takes the hidden state
        # and outputs a probability of 'continue looping'
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size // 4)
        self.dense2 = nn.Linear(config.hidden_size // 4, 1)
        self.act = nn.GELU()

    def forward(self, hidden_states):
        # We pool the sequence (e.g., take the last token's state representing the current reasoning step)
        # hidden_states: [batch_size, seq_len, hidden_size]
        pooled_state = hidden_states[:, -1, :] # [batch_size, hidden_size]
        
        x = self.act(self.dense1(pooled_state))
        continue_logits = self.dense2(x) # [batch_size, 1]
        
        # Sigmoid gives probability between 0 and 1
        continue_prob = torch.sigmoid(continue_logits)
        return continue_prob

class Attention(nn.Module):
    def __init__(self, config: ALRConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.hidden_size, dim=2)
        
        q = q.view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        # Causal mask - standard lower triangular
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        
        # standard scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        if attention_mask is not None:
             # Apply additional custom mask if provided
             att = att + attention_mask
             
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.o_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config: ALRConfig):
        super().__init__()
        # Using GeGLU or standard MLP (here standard for simplicity in prototype, 
        # could be optimized to SwiGLU for better efficiency)
        self.c_fc    = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.c_proj  = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act     = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))

class TransformerBlock(nn.Module):
    def __init__(self, config: ALRConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class ALRModel(nn.Module):
    """
    Adaptive Latent Recurrence Model (ALR-DKR)
    """
    def __init__(self, config: ALRConfig):
        super().__init__()
        self.config = config
        
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        
        # Language Modeling Head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Weight tying
        self.token_embeddings.weight = self.lm_head.weight
        
        # The Innovation: The Latent Recurrence Gate
        self.recurrence_gate = LatentRecurrenceGate(config)

    def get_num_params(self):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _forward_blocks(self, hidden_states, attention_mask=None):
        """Pass through the physical layers once."""
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states

    def forward(self, input_ids, attention_mask=None, require_recurrence_stats=False):
        B, T = input_ids.size()
        device = input_ids.device
        
        # 1. Initial Embedding
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        tok_emb = self.token_embeddings(input_ids) # shape (b, t, n_embd)
        pos_emb = self.position_embeddings(pos) # shape (1, t, n_embd)
        
        hidden_states = tok_emb + pos_emb
        
        # 2. Adaptive Latent Recurrence Loop
        # In a generic batch processing forward pass during training, we might just unroll this
        # or compute expected recurrence. For this prototype inference/forward pass, we simulate the loop.
        
        recurrence_counts = torch.zeros(B, 1, device=device)
        continue_probs_history = []
        
        for step in range(self.config.max_recurrence_steps):
            # Pass through the physical architecture
            hidden_states = self._forward_blocks(hidden_states, attention_mask=attention_mask)
            
            # Evaluate if we need to loop again
            continue_prob = self.recurrence_gate(hidden_states) # [B, 1]
            continue_probs_history.append(continue_prob)
            
            # If all sequences in the batch fall below the threshold, break early
            # (In a real batched inference scenario, we'd mask out the ones that are done
            # and only update the hidden states of sequences still "thinking")
            if (continue_prob < self.config.recurrence_threshold).all() and step > 0:
                break
            
            # For logging/stats
            recurrence_counts += (continue_prob >= self.config.recurrence_threshold).float()
            
            # Soft gating: blend the state. This makes it differentiable during training.
            # Next state = p * f(h) + (1-p) * h
            # For stability in training, we can blend it. For hard gating in inference, we'd just loop.
            # We'll use soft gating here so the model learns 'when to stop'.
            # Note: For strict prototype testing, we will just proceed with the updated hidden_states.
            
        # 3. Final Output Projection
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states) # (B, T, vocab_size)
        
        return logits, recurrence_counts, continue_probs_history
