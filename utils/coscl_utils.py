"""
CoSCL utilities: Fisher matrix computation and KLD loss
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class KLD(nn.Module):
    """
    KL Divergence loss for expert cooperation in CoSCL.
    Encourages experts to have similar output distributions.
    """
    def __init__(self):
        super(KLD, self).__init__()
        self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x):
        """
        Compute KL divergence between all pairs of expert outputs.
        
        Args:
            x: List of expert logits [expert1_logits, expert2_logits, ...]
        
        Returns:
            Total KL divergence loss
        """
        KLD_loss = 0
        for k in range(len(x)):
            for l in range(len(x)):
                if l != k:
                    KLD_loss += self.criterion_KLD(
                        F.log_softmax(x[k], dim=1), 
                        F.softmax(x[l], dim=1).detach()
                    )
        return KLD_loss


def fisher_matrix_diag_coscl(t, x, y, model, criterion, sbatch=20, device=None):
    """
    Compute Fisher Information Matrix diagonal for CoSCL.
    
    Args:
        t: Task index
        x: Input data tensor
        y: Target labels tensor
        model: CoSCL model
        criterion: Loss criterion
        sbatch: Batch size for computation
        device: Device to use (if None, uses model's device)
    
    Returns:
        Dictionary of Fisher information for each parameter
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Initialize Fisher matrix
    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = torch.zeros_like(p.data)
        p.requires_grad_()
    
    # Compute Fisher information
    model.train()
    criterion_loss = nn.CrossEntropyLoss()
    
    num_samples = x.size(0)
    num_batches = (num_samples + sbatch - 1) // sbatch
    
    for i in tqdm(range(0, num_samples, sbatch), desc='Fisher diagonal', ncols=100, ascii=True):
        end_idx = min(i + sbatch, num_samples)
        b = torch.arange(i, end_idx).to(device)
        images = x[b].to(device)
        targets = y[b].to(device)
        task = torch.LongTensor([t]).to(device)

        # Forward and backward
        model.zero_grad()
        outputs = model.forward(images, task)
        
        # Ensure targets are within valid range for the output layer
        num_classes = outputs.size(1)
        targets = torch.clamp(targets, 0, num_classes - 1)
        
        loss = criterion_loss(outputs, targets)
        loss.backward()
        
        # Accumulate squared gradients
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += (end_idx - i) * p.grad.data.pow(2)
    
    # Normalize by number of samples
    with torch.no_grad():
        for n, _ in model.named_parameters():
            fisher[n] = fisher[n] / num_samples
    
    return fisher

