"""
Yeast promoter MPRA dataset loader.

Dataset from: de Boer et al., Nature Biotechnology 2024
Zenodo: https://zenodo.org/records/10633252

Following DREAM challenge preprocessing:
- 80bp random sequences padded to 150bp with plasmid context
- 6 channels: ACGT + reverse complement flag + singleton flag
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

from .base import SequenceDataset
from .utils import one_hot_encode


class YeastDataset(SequenceDataset):
    """
    Yeast promoter MPRA dataset.
    
    Dataset characteristics:
    - 80bp random promoter sequences
    - Padded to 150bp with plasmid context (57bp + 80bp + 13bp)
    - 6 input channels: ACGT + reverse complement flag + singleton flag
    - Expression values (continuous)
    
    Data splits (following the paper):
    - train: 100,000 sequences (for active learning experiments)
    - pool: 5,965,324 sequences (remaining training data for selection)
    - val: 20,000 sequences (subset of validation set)
    - test: 71,103 sequences (DREAM challenge test sets)
    """
    
    # Plasmid flanking sequences from DREAM challenge (AddGene plasmid 127546)
    # 5' flank includes: distal region (54bp) + TCG from XhoI restriction site (3bp) = 57bp
    FLANK_5_PRIME = "GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATCTCG"  # 57bp
    FLANK_3_PRIME = "GGTTACGGCTGTT"  # 13bp (first 13bp of proximal region)
    
    SEQUENCE_LENGTH = 150  # Total length after padding: 57 + 80 + 13
    RANDOM_REGION_LENGTH = 80  # Length of random promoter region
    NUM_CHANNELS = 6  # ACGT + reverse complement flag + singleton flag
    
    def __init__(self, data_path: str, split: str = 'train', subset_size: Optional[int] = None):
        """
        Initialize yeast dataset.
        
        Args:
            data_path: Path to data directory
            split: One of 'train', 'pool', 'val', 'test'
            subset_size: Optional number of samples to use (for downsampling experiments)
        """
        self.subset_size = subset_size
        super().__init__(data_path, split)
    
    def load_data(self) -> None:
        """
        Load yeast MPRA data with proper train/pool/val/test splits.
        
        Following the paper:
        - train.txt contains full training data (~6.06M)
        - Split into: 100K train + 5.9M pool
        - val.txt contains full validation data (~673K) 
        - Use random 20K subset for validation
        - test: Use filtered_test_data_with_MAUDE_expression.txt
        """
        data_dir = Path(self.data_path)
        
        # Handle different splits
        if self.split in ['train', 'pool']:
            file_path = data_dir / 'train.txt'
            print(f"Loading Yeast train data from {file_path}")
            df = pd.read_csv(file_path, sep='\t', header=None, names=['sequence', 'expression'])
            
            # Check for singleton flag (integer vs float expression values)
            is_singleton = (df['expression'] % 1 == 0).values.astype(np.float32)
            
            # For baseline experiments: use full training set
            # For active learning: split into train (first 100K) and pool (rest)
            if self.split == 'pool':
                # Pool split only used for active learning
                # Use truly random permutation
                indices = np.random.permutation(len(df))
                indices = indices[100000:]
                df = df.iloc[indices].reset_index(drop=True)
                is_singleton = is_singleton[indices]
                print(f"Using remaining {len(indices):,} sequences for pool (random permutation)")
            else:
                # For 'train' split: use full dataset (6.7M sequences)
                # Downsampling will be applied later if subset_size is specified
                print(f"Using full training set: {len(df):,} sequences")
            
        elif self.split == 'val':
            file_path = data_dir / 'val.txt'
            print(f"Loading Yeast val data from {file_path}")
            df = pd.read_csv(file_path, sep='\t', header=None, names=['sequence', 'expression'])
            
            # Use random 20K subset for validation (per paper)
            # Use truly random sampling without replacement
            if len(df) > 20000:
                indices = np.random.choice(len(df), size=20000, replace=False)
                df = df.iloc[indices].reset_index(drop=True)
                print(f"Using random 20,000 subset of validation set (random sampling, no replacement)")
            
            is_singleton = (df['expression'] % 1 == 0).values.astype(np.float32)
            
        elif self.split == 'test':
            file_path = data_dir / 'filtered_test_data_with_MAUDE_expression.txt'
            print(f"Loading Yeast test data from {file_path}")
            df = pd.read_csv(file_path, sep='\t', header=None, names=['sequence', 'expression'])
            is_singleton = (df['expression'] % 1 == 0).values.astype(np.float32)
        else:
            raise ValueError(f"Invalid split: {self.split}. Expected one of: train, pool, val, test")
        
        # Extract sequences and labels
        self.sequences = df['sequence'].values
        self.labels = df['expression'].values.astype(np.float32)
        self.is_singleton = is_singleton
        
        # Add plasmid flanking regions to get 150bp sequences
        self.sequences = self._add_plasmid_context(self.sequences)
        
        # Apply subset size if specified (for downsampling experiments)
        # Use truly random sampling without replacement
        if self.subset_size is not None and self.subset_size < len(self.sequences):
            indices = np.random.choice(len(self.sequences), size=self.subset_size, replace=False)
            self.sequences = self.sequences[indices]
            self.labels = self.labels[indices]
            self.is_singleton = self.is_singleton[indices]
            print(f"Downsampled to {self.subset_size} sequences (random sampling, no replacement)")
        
        self.sequence_length = self.SEQUENCE_LENGTH
        
        print(f"Loaded {len(self.sequences)} sequences for {self.split} split")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Label range: [{np.min(self.labels):.3f}, {np.max(self.labels):.3f}]")
    
    def _add_plasmid_context(self, sequences: np.ndarray) -> np.ndarray:
        """
        Add plasmid flanking sequences to get 150bp sequences.
        
        Current sequences are ~109-110bp with partial flanking:
        - Last 17bp of 5' flank: TGCATTTTTTTCACATC
        - Random region: ~79-80bp
        - Full 3' flank: GGTTACGGCTGTT (13bp)
        
        We add the full 57bp 5' flank which includes:
        - 54bp distal region
        - 3bp from XhoI restriction site (TCG from TCGAGG)
        
        Final structure: [57bp 5' flank] + [80bp random] + [13bp 3' flank] = 150bp
        """
        processed = []
        partial_5_prime = self.FLANK_5_PRIME[-17:]  # Last 17bp of 5' flank (TGCATTTTTTTCACATC)
        
        for seq in sequences:
            # Remove 3' flank if present (last 13bp)
            if seq.endswith(self.FLANK_3_PRIME):
                seq = seq[:-len(self.FLANK_3_PRIME)]
            
            # Remove partial 5' flank if present (first 17bp)
            if seq.startswith(partial_5_prime):
                seq = seq[len(partial_5_prime):]
            
            # Now seq should be the random region (~80bp)
            # Pad or truncate to exactly 80bp
            if len(seq) < self.RANDOM_REGION_LENGTH:
                # Pad with N (right side)
                seq = seq + 'N' * (self.RANDOM_REGION_LENGTH - len(seq))
            elif len(seq) > self.RANDOM_REGION_LENGTH:
                # Truncate (keep first 80bp)
                seq = seq[:self.RANDOM_REGION_LENGTH]
            
            # Add full plasmid flanking sequences
            full_seq = self.FLANK_5_PRIME + seq + self.FLANK_3_PRIME
            
            # Verify length
            if len(full_seq) != self.SEQUENCE_LENGTH:
                raise ValueError(
                    f"Sequence length mismatch: {len(full_seq)} != {self.SEQUENCE_LENGTH}\n"
                    f"5' flank: {len(self.FLANK_5_PRIME)}, random: {len(seq)}, 3' flank: {len(self.FLANK_3_PRIME)}"
                )
            
            processed.append(full_seq)
        
        return np.array(processed)
    
    def encode_sequence(self, sequence: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Encode a yeast sequence with 6 channels.
        
        Args:
            sequence: DNA sequence string (150bp)
            metadata: Optional metadata dict with 'is_singleton' flag
            
        Returns:
            Encoded sequence of shape (6, 150)
            Channels:
            - 0-3: one-hot encoded ACGT
            - 4: reverse complement flag (0 for forward, 1 for reverse)
            - 5: singleton flag (1 for integer expression values)
        """
        # Get one-hot encoding (4 channels)
        encoded = one_hot_encode(sequence, add_singleton_channel=False)  # Shape: (4, 150)
        
        # Add reverse complement channel (always 0 for forward strand during training)
        rc_channel = np.zeros((1, len(sequence)), dtype=np.float32)
        
        # Add singleton channel
        if metadata is not None and 'is_singleton' in metadata:
            singleton_value = metadata['is_singleton']
        else:
            singleton_value = 0.0  # Default: not singleton
        
        singleton_channel = np.full((1, len(sequence)), singleton_value, dtype=np.float32)
        
        # Concatenate all channels: (4, 150) + (1, 150) + (1, 150) = (6, 150)
        encoded = np.concatenate([encoded, rc_channel, singleton_channel], axis=0)
        
        return encoded
    
    def __getitem__(self, idx: int) -> tuple:
        """Get a single sample."""
        import torch
        
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Create metadata dict with singleton flag
        metadata = {'is_singleton': self.is_singleton[idx] if hasattr(self, 'is_singleton') else 0.0}
        
        # Encode sequence
        encoded = self.encode_sequence(sequence, metadata)
        
        # Convert to torch tensors
        encoded_tensor = torch.from_numpy(encoded).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return encoded_tensor, label_tensor
    
    def get_num_channels(self) -> int:
        """Return number of input channels (6 for yeast)."""
        return self.NUM_CHANNELS
