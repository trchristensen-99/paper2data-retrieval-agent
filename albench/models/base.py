"""
Abstract base class for sequence-to-function models.

This module provides a common interface for models that predict
functional measurements from genomic sequences.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class SequenceModel(ABC, nn.Module):
    """
    Abstract base class for sequence-to-function prediction models.
    
    Provides a common interface for models that take one-hot encoded
    DNA sequences as input and predict continuous functional values.
    
    Attributes:
        input_channels: Number of input channels (e.g., 4 for ACGT, 5 for ACGT+metadata)
        sequence_length: Expected sequence length
        output_dim: Output dimension (typically 1 for regression)
    """
    
    def __init__(
        self,
        input_channels: int,
        sequence_length: int,
        output_dim: int = 1
    ):
        """
        Initialize model.
        
        Args:
            input_channels: Number of input channels
            sequence_length: Expected sequence length
            output_dim: Output dimension (default: 1 for regression)
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        pass
    
    def predict(
        self, 
        x: torch.Tensor, 
        use_reverse_complement: bool = True
    ) -> torch.Tensor:
        """
        Make predictions, optionally averaging with reverse complement.
        
        This is important for genomic sequences since regulatory activity
        should ideally be orientation-independent.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length)
               For yeast: (batch, 6, 150) - ACGT + RC flag + singleton flag
               For K562: (batch, 5, 200) - ACGT + RC flag
            use_reverse_complement: Whether to average predictions from both orientations
            
        Returns:
            Predicted values of shape (batch_size, output_dim)
        """
        self.eval()
        
        with torch.no_grad():
            # Forward prediction
            pred_fwd = self.forward(x)
            
            if not use_reverse_complement:
                return pred_fwd
            
            # Reverse complement prediction
            # 1. Reverse along sequence dimension
            x_rc = x.flip(dims=[2])
            
            # 2. Swap ACGT channels: A<->T (0<->3), C<->G (1<->2)
            x_rc_swapped = x_rc.clone()
            x_rc_swapped[:, [0, 1, 2, 3], :] = x_rc[:, [3, 2, 1, 0], :]
            
            # 3. Set reverse complement flag to 1 (channel 4)
            if x.shape[1] >= 5:  # Has RC flag channel
                x_rc_swapped[:, 4, :] = 1.0
            
            # 4. Keep other metadata channels unchanged (e.g., singleton flag)
            # They're already copied in the clone() operation
            
            pred_rc = self.forward(x_rc_swapped)
            
            # Average predictions
            return (pred_fwd + pred_rc) / 2.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model architecture information.
        
        Returns:
            Dictionary with model metadata
        """
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": self.__class__.__name__,
            "input_channels": self.input_channels,
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
            "num_parameters": num_params,
            "num_trainable_parameters": num_trainable,
        }
    
    def save_checkpoint(self, path: str, **kwargs) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            **kwargs: Additional metadata to save
        """
        # Ensure parent directory exists
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            **kwargs
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, strict: bool = True) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            strict: Whether to strictly enforce state_dict keys match
            
        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        return checkpoint
