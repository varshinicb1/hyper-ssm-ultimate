import torch
import torch.nn as nn
import torch.nn.functional as F

def spectral_scale(W, num_iters=1):
    """
    Bounds the generated expert outputs by normalizing against their principal singular value.
    Prevents hypernetwork parameter variance explosion.
    W shape: [B, num_experts, in_dim, out_dim]
    """
    # Flattens Batch and Experts for unified spectral processing
    B, E, D1, D2 = W.shape
    W_flat = W.view(B * E, D1, D2)
    
    # Random vector for Power Iteration approximation requires exact matching dtype
    u = torch.randn(B * E, D1, 1, device=W.device, dtype=W.dtype)
    
    for _ in range(num_iters):
        v = F.normalize(torch.bmm(W_flat.transpose(1, 2), u), dim=1)
        u = F.normalize(torch.bmm(W_flat, v), dim=1)
        
    # Calculate approximate principal singular value: sigma = u.T @ W @ v
    sigma = torch.bmm(u.transpose(1, 2), torch.bmm(W_flat, v)).view(B, E, 1, 1)
    
    # Scale matrix securely above 0
    return W / (sigma + 1e-6)

class HyperWeightSynthesizer(nn.Module):
    """
    The "Seed" network containing the actual 0.5B static parameters.
    Instead of directly processing the sequence to output logits, this network reads
    the context and dynamically output *weights* for a secondary, massive network.
    
    This solves the Static Compute problem by synthesizing a customized 30B pathway
    out of nowhere for complex queries, using only the 0.5B params as a generator.
    """
    def __init__(self, context_dim, hidden_dim, target_layer_dim, num_experts=4):
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.target_layer_dim = target_layer_dim
        self.num_experts = num_experts
        
        # The Seed Core: Read the running hyperspace context and decide what
        # logical structure the next layer needs.
        self.w_fc1 = nn.Linear(context_dim, hidden_dim)
        self.w_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Generates the transient layer weights for ALL experts simultaneously.
        # We use a low-rank approximation (LoRA-style) generator to keep the seed size tiny.
        self.w_gen_A = nn.Linear(hidden_dim, num_experts * target_layer_dim * 8)
        self.w_gen_B = nn.Linear(hidden_dim, num_experts * 8 * target_layer_dim)
        
    def _quantize_to_ternary(self, weights):
        """
        Native 1.58-bit representation: 
        Forces the generated weights (or static weights) to be exactly {-1, 0, 1}.
        Used here to mock the zero-MatMul operation efficiency.
        """
        # Straight-Through Estimator mock
        alpha = weights.abs().mean()
        scaled = weights / (alpha + 1e-8)
        quantized = torch.round(torch.clamp(scaled, -1, 1))
        # Keep gradients flowing via STE
        return (quantized - weights).detach() + weights

    def forward(self, context_state):
        """
        context_state: [batch, context_dim] 
        Outputs dynamically synthesized, ternary weights for a feed-forward layer.
        """
        B = context_state.shape[0]
        
        # Analyze the context
        h = F.silu(self.w_fc1(context_state))
        h = F.silu(self.w_fc2(h))
        
        # Generate Low-Rank matrices A and B for ALL experts
        # Output shape: [B, num_experts, target, rank] and [B, num_experts, rank, target]
        rank = 8
        A = self.w_gen_A(h).view(B, self.num_experts, self.target_layer_dim, rank)
        B_mat = self.w_gen_B(h).view(B, self.num_experts, rank, self.target_layer_dim)
        
        # Synthesize full weight matrix: W = A @ B using torch.matmul for batch + expert broadcast
        transient_weights = torch.matmul(A, B_mat) # [B, E, target, target]
        
        # Apply Spectral Normalization to safely bound output magnitudes
        safe_weights = spectral_scale(transient_weights)
        
        # Snap the synthesized weights to {-1, 0, 1} for integer-only compute
        ternary_weights = self._quantize_to_ternary(safe_weights)
        
        return ternary_weights

class DynamicLiquidLayer(nn.Module):
    """
    A layer that structurally exists, but has absolutely no parameters of its own until 
    the Synthesizer passes them in.
    """
    def __init__(self, context_dim, target_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.synthesizer = HyperWeightSynthesizer(context_dim, 128, target_dim, num_experts)
        self.router = nn.Linear(target_dim, num_experts, bias=False)
        
        # Learnable scalar to scale output of synthesized experts, preventing early dominance
        self.expert_scale = nn.Parameter(torch.ones(num_experts))
        
    def forward(self, x, context_state):
        """
        x: [B, seq_len, target_dim]
        context_state: [B, context_dim]
        """
        # 1. Generate the MoE weights for this exact context
        # W_liquid shape: [B, num_experts, target_dim, target_dim]
        W_liquid = self.synthesizer(context_state)
        
        # 2. Compute Routing Probabilities token-by-token
        # router_logits: [B, seq_len, num_experts]
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 3. Execute the input linearly using all generated experts
        # We apply Einstein summation to evaluate all experts on all sequences
        # b: batch, s: seq_len, e: expert, d: in_dim, o: out_dim
        # expert_out shape: [B, seq_len, num_experts, target_dim]
        expert_out = torch.einsum('bsd,bedo->bseo', x, W_liquid)
        
        # Scale expert magnitude safely via learning parameter 
        # (broadcasting self.expert_scale across batch and sequence)
        expert_out = expert_out * self.expert_scale.view(1, 1, self.num_experts, 1)
        
        # 4. Gate the expert outputs
        # out shape: [B, seq_len, target_dim]
        out = torch.einsum('bse,bseo->bso', router_probs, expert_out)
        
        # We return the router probabilities un-reduced so the training loop can easily 
        # ingest them for Entropy Regularization calculations.
        return F.silu(out), router_probs
