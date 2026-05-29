import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalKnowledgeLoss(nn.Module):
    """
    Simulates a loss function that penalizes the model for storing static factual
    knowledge in its parameters. By minimizing this loss, the model is pressured
    to rely strictly on logical routing and external contexts for factual retrieval.
    """
    def __init__(self, penalty_weight=0.01):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(self, logits, target, model_params):
        """
        logits: (batch_size, seq_len, vocab_size)
        target: (batch_size, seq_len)
        model_params: list of parameters to compute explicit regularization on
        """
        # standard next-token prediction loss
        # Flatten the logits and targets
        loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        
        # Orthogonality Penalty:
        # In this prototype, we simulate the penalization of "memorizing specific entity tokens".
        # A true implementation would have an external vector DB signal.
        # Here we apply an L2-style regularization strongly to the final MLPs 
        # to prevent parameters from becoming 'dense' with facts.
        
        reg_loss = 0.0
        for p in model_params:
            if p.requires_grad:
                # Penalize large absolute weights to force distributed, generalizable representations
                reg_loss += torch.norm(p, p=2)
        
        total_loss = loss_ce + (self.penalty_weight * reg_loss)
        return total_loss, loss_ce.item(), reg_loss.item()
