"""
Utility functions for loss computation, especially for yeast bin-based classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def continuous_to_bin_probabilities(
    continuous_values: torch.Tensor,
    num_bins: int = 18,
    min_val: float = 0.0,
    max_val: float = 17.0
) -> torch.Tensor:
    """
    Convert continuous expression values to bin probability distributions.
    
    For yeast: expression values are integers 0-17, so we use one-hot encoding.
    For non-integer values, we use soft assignment to nearest bins.
    
    Args:
        continuous_values: Continuous expression values of shape (batch_size,)
        num_bins: Number of bins (18 for yeast)
        min_val: Minimum expression value (0 for yeast)
        max_val: Maximum expression value (17 for yeast)
        
    Returns:
        Bin probability distributions of shape (batch_size, num_bins)
    """
    batch_size = continuous_values.shape[0]
    device = continuous_values.device
    
    # Clamp values to valid range
    continuous_values = torch.clamp(continuous_values, min_val, max_val)
    
    # For integer values (yeast case), use one-hot encoding
    # Round to nearest integer
    integer_values = torch.round(continuous_values).long()
    
    # Create one-hot encoding
    bin_probs = torch.zeros(batch_size, num_bins, device=device)
    bin_probs.scatter_(1, integer_values.unsqueeze(1), 1.0)
    
    return bin_probs


class YeastKLLoss(nn.Module):
    """
    KL divergence loss for yeast 18-bin classification.
    
    This loss:
    1. Gets logits from the model (before SoftMax)
    2. Converts continuous labels to bin probabilities
    3. Applies log_softmax to logits
    4. Computes KL divergence
    """
    
    def __init__(self, reduction: str = 'batchmean'):
        super().__init__()
        self.reduction = reduction
        self.kl_div = nn.KLDivLoss(reduction=reduction)
    
    def forward(
        self,
        model_logits: torch.Tensor,
        continuous_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence loss.
        
        Args:
            model_logits: Raw logits from model of shape (batch_size, 18)
            continuous_targets: Continuous expression values of shape (batch_size,)
            
        Returns:
            KL divergence loss
        """
        # Convert continuous targets to bin probabilities
        target_probs = continuous_to_bin_probabilities(
            continuous_targets,
            num_bins=18,
            min_val=0.0,
            max_val=17.0
        )
        
        # Apply log_softmax to logits (required for KL divergence)
        log_probs = F.log_softmax(model_logits, dim=1)
        
        # Compute KL divergence: KL(target || predicted)
        loss = self.kl_div(log_probs, target_probs)
        
        return loss
