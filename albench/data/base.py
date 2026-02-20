"""
Abstract base class for genomic sequence datasets.

This module provides a common interface for loading and preprocessing
MPRA datasets with sequence-to-function mappings.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(ABC, Dataset):
    """
    Abstract base class for genomic sequence datasets.
    
    This class provides a common interface for loading and preprocessing
    MPRA datasets with sequence-to-function mappings. Subclasses should
    implement data loading and preprocessing specific to their dataset.
    
    Attributes:
        data_path: Path to data directory
        split: Which split to load ("train", "val", or "test")
        sequences: Array of DNA sequences (strings)
        labels: Array of functional measurements (continuous values)
        metadata: Additional information about sequences
        sequence_length: Expected length of sequences
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to data directory
            split: Which split to load ("train", "val", or "test")
            transform: Optional transform to apply to sequences
            target_transform: Optional transform to apply to labels
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")
        
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # These will be populated by load_data()
        self.sequences: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.sequence_length: Optional[int] = None
        
        # Load data
        self.load_data()
        
        if self.sequences is None or self.labels is None:
            raise RuntimeError("load_data() must populate sequences and labels")
        
        if len(self.sequences) != len(self.labels):
            raise ValueError(
                f"Sequences and labels must have same length, got "
                f"{len(self.sequences)} sequences and {len(self.labels)} labels"
            )
    
    @abstractmethod
    def load_data(self) -> None:
        """
        Load sequences and labels from source.
        
        This method should populate:
        - self.sequences: Array of DNA sequences (strings)
        - self.labels: Array of continuous values
        - self.metadata: Dict with additional info (optional)
        - self.sequence_length: Expected sequence length
        """
        pass
    
    @abstractmethod
    def encode_sequence(self, sequence: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Encode a single sequence into numerical format.
        
        Args:
            sequence: DNA sequence string
            metadata: Optional metadata for this specific sequence
            
        Returns:
            Encoded sequence as numpy array
        """
        pass
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of sample to retrieve
            
        Returns:
            Tuple of (encoded_sequence, label) as torch tensors
        """
        # Get sequence and label
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Get metadata if available
        sample_metadata = None
        if self.metadata is not None:
            sample_metadata = {key: val[idx] for key, val in self.metadata.items()}
        
        # Encode sequence
        encoded_seq = self.encode_sequence(sequence, sample_metadata)
        
        # Convert to tensors
        seq_tensor = torch.from_numpy(encoded_seq).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        # Apply transforms if specified
        if self.transform is not None:
            seq_tensor = self.transform(seq_tensor)
        if self.target_transform is not None:
            label_tensor = self.target_transform(label_tensor)
        
        return seq_tensor, label_tensor
    
    def get_sequence_length(self) -> int:
        """Get the sequence length for this dataset."""
        if self.sequence_length is None:
            raise ValueError("sequence_length not set")
        return self.sequence_length
    
    def get_num_channels(self) -> int:
        """
        Get the number of input channels for this dataset.
        
        Should be overridden by subclasses that use different numbers of channels.
        """
        return 4  # Default: ACGT
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics about the dataset
        """
        return {
            "num_samples": len(self),
            "sequence_length": self.sequence_length,
            "num_channels": self.get_num_channels(),
            "label_mean": float(np.mean(self.labels)),
            "label_std": float(np.std(self.labels)),
            "label_min": float(np.min(self.labels)),
            "label_max": float(np.max(self.labels)),
        }
    
    def create_subset(self, indices: np.ndarray) -> "SequenceDataset":
        """
        Create a subset of this dataset with specified indices.
        
        Args:
            indices: Array of indices to include in subset
            
        Returns:
            New dataset instance with only specified samples
            
        Note:
            This creates a shallow copy with views into the original arrays.
        """
        # Create a new instance (this is a bit hacky but works)
        subset = self.__class__.__new__(self.__class__)
        subset.data_path = self.data_path
        subset.split = self.split
        subset.transform = self.transform
        subset.target_transform = self.target_transform
        subset.sequence_length = self.sequence_length
        
        # Create views with subset of data
        subset.sequences = self.sequences[indices]
        subset.labels = self.labels[indices]
        
        if self.metadata is not None:
            subset.metadata = {key: val[indices] for key, val in self.metadata.items()}
        else:
            subset.metadata = None
        
        return subset
