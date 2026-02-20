"""
Data pool management for active learning.

Manages the partition between labeled training data and unlabeled pool data,
ensuring no duplicates and providing clean interfaces for AL experiments.
"""

from typing import Optional, Set, List
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class DataPool:
    """
    Manages labeled and unlabeled data partitions for active learning.
    
    This class maintains the split between:
    - Labeled set: Data that has been "acquired" and can be used for training
    - Pool set: Unlabeled data available for selection by acquisition strategies
    
    Ensures no overlap between sets and tracks acquisition history.
    """
    
    def __init__(
        self,
        full_dataset: Dataset,
        initial_labeled_indices: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize data pool.
        
        Args:
            full_dataset: Complete dataset to partition
            initial_labeled_indices: Initial labeled samples (if None, start empty)
            random_seed: Random seed for reproducibility
        """
        self.full_dataset = full_dataset
        self.dataset_size = len(full_dataset)
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize labeled and unlabeled sets
        if initial_labeled_indices is not None:
            self.labeled_indices = set(initial_labeled_indices.tolist())
        else:
            self.labeled_indices = set()
        
        # Pool contains all indices not in labeled set
        self.unlabeled_indices = set(range(self.dataset_size)) - self.labeled_indices
        
        # Track acquisition history
        self.acquisition_history = []
        if initial_labeled_indices is not None:
            self.acquisition_history.append({
                'round': 0,
                'indices': initial_labeled_indices.tolist(),
                'num_acquired': len(initial_labeled_indices)
            })
    
    def add_to_labeled(self, indices: np.ndarray, round_num: Optional[int] = None) -> None:
        """
        Move samples from pool to labeled set.
        
        Args:
            indices: Indices to move from pool to labeled set
            round_num: AL round number for tracking
            
        Raises:
            ValueError: If any index is already labeled or out of bounds
        """
        indices_set = set(indices.tolist())
        
        # Validate indices
        if not indices_set.issubset(self.unlabeled_indices):
            already_labeled = indices_set - self.unlabeled_indices
            if already_labeled:
                raise ValueError(
                    f"Attempting to add already labeled indices: {already_labeled}"
                )
        
        # Move from pool to labeled
        self.labeled_indices.update(indices_set)
        self.unlabeled_indices.difference_update(indices_set)
        
        # Record in history
        self.acquisition_history.append({
            'round': round_num if round_num is not None else len(self.acquisition_history),
            'indices': indices.tolist(),
            'num_acquired': len(indices)
        })
    
    def get_labeled_dataset(self) -> Subset:
        """
        Get dataset containing only labeled samples.
        
        Returns:
            Subset of full dataset with labeled indices
        """
        return Subset(self.full_dataset, sorted(self.labeled_indices))
    
    def get_pool_dataset(self) -> Subset:
        """
        Get dataset containing only unlabeled samples (the pool).
        
        Returns:
            Subset of full dataset with unlabeled indices
        """
        return Subset(self.full_dataset, sorted(self.unlabeled_indices))
    
    def get_pool_indices(self) -> np.ndarray:
        """Get array of unlabeled indices."""
        return np.array(sorted(self.unlabeled_indices))
    
    def get_labeled_indices(self) -> np.ndarray:
        """Get array of labeled indices."""
        return np.array(sorted(self.labeled_indices))
    
    def num_labeled(self) -> int:
        """Number of labeled samples."""
        return len(self.labeled_indices)
    
    def num_unlabeled(self) -> int:
        """Number of unlabeled samples in pool."""
        return len(self.unlabeled_indices)
    
    def is_exhausted(self) -> bool:
        """Check if pool is empty."""
        return len(self.unlabeled_indices) == 0
    
    def get_statistics(self) -> dict:
        """
        Get statistics about current pool state.
        
        Returns:
            Dictionary with pool statistics
        """
        return {
            'total_samples': self.dataset_size,
            'num_labeled': self.num_labeled(),
            'num_unlabeled': self.num_unlabeled(),
            'fraction_labeled': self.num_labeled() / self.dataset_size,
            'num_acquisition_rounds': len(self.acquisition_history),
            'is_exhausted': self.is_exhausted()
        }
    
    def reset(self, initial_labeled_indices: Optional[np.ndarray] = None) -> None:
        """
        Reset pool to initial state.
        
        Args:
            initial_labeled_indices: New initial labeled set (if None, start empty)
        """
        if initial_labeled_indices is not None:
            self.labeled_indices = set(initial_labeled_indices.tolist())
        else:
            self.labeled_indices = set()
        
        self.unlabeled_indices = set(range(self.dataset_size)) - self.labeled_indices
        self.acquisition_history = []
        
        if initial_labeled_indices is not None:
            self.acquisition_history.append({
                'round': 0,
                'indices': initial_labeled_indices.tolist(),
                'num_acquired': len(initial_labeled_indices)
            })
    
    def __repr__(self) -> str:
        return (
            f"DataPool(total={self.dataset_size}, "
            f"labeled={self.num_labeled()}, "
            f"unlabeled={self.num_unlabeled()})"
        )
