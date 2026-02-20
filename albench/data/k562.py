"""
K562 human cell line MPRA dataset loader.

Dataset from: Gosai et al., Nature 2023
Zenodo: https://zenodo.org/records/10698014

Following benchmark paper preprocessing:
- 200bp genomic sequences (pad shorter sequences with Ns)
- 5 channels: ACGT + reverse complement flag
- hashFrag-based orthogonal train/val/test splits
"""

import os
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

from .base import SequenceDataset
from .utils import one_hot_encode
from .hashfrag_splits import HashFragSplitter

logger = logging.getLogger(__name__)


class K562Dataset(SequenceDataset):
    """
    K562 human MPRA dataset.
    
    Dataset characteristics:
    - 367,364 regulatory sequences (reference alleles only)
    - 200bp genomic sequences
    - 5 input channels: ACGT + reverse complement flag
    - Expression values (log2 fold change)
    
    Data splits (following the paper):
    - train: 100,000 sequences (for active learning experiments)
    - pool: 193,890 sequences (remaining training data for selection)
    - val: 36,737 sequences (hashFrag-based validation set)
    - test: 36,737 sequences (hashFrag-based test set)
    """
    
    SEQUENCE_LENGTH = 200  # Target sequence length (as per paper)
    NUM_CHANNELS = 5  # ACGT + reverse complement flag
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[any] = None,
        target_transform: Optional[any] = None,
        subset_size: Optional[int] = None,
        use_hashfrag: bool = True,
        hashfrag_threshold: int = 60,
        hashfrag_cache_dir: Optional[str] = None,
        use_chromosome_fallback: bool = False
    ):
        """
        Initialize K562 dataset.
        
        Args:
            data_path: Path to data directory containing the main data file
            split: One of 'train', 'pool', 'val', 'test'
            transform: Optional transform to apply to sequences
            target_transform: Optional transform to apply to labels
            subset_size: Optional number of samples to use (for downsampling experiments)
            use_hashfrag: If True, use HashFrag for orthogonal splits (default: True)
            hashfrag_threshold: Smith-Waterman score threshold for homology (default: 60)
            hashfrag_cache_dir: Directory to cache HashFrag splits (default: {data_path}/hashfrag_splits)
            use_chromosome_fallback: If True, use chromosome-based splits as fallback (default: False)
        """
        self.subset_size = subset_size
        self.use_hashfrag = use_hashfrag
        self.hashfrag_threshold = hashfrag_threshold
        self.hashfrag_cache_dir = hashfrag_cache_dir
        self.use_chromosome_fallback = use_chromosome_fallback
        super().__init__(data_path, split, transform, target_transform)
    
    def load_data(self) -> None:
        """
        Load K562 MPRA data with hashFrag-based train/pool/val/test splits.
        
        Following the paper:
        - Use hashFrag to generate orthogonal splits (80:10:10)
        - From 80% training data: 100K train + 193K pool
        - 10% validation (36,737 sequences)
        - 10% test (36,737 sequences)
        """
        data_dir = Path(self.data_path)
        
        # The actual filename from the Zenodo download
        file_path = data_dir / 'DATA-Table_S2__MPRA_dataset.txt'
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Could not find K562 data file at {file_path}. "
                f"Please run: python scripts/download_data.py --dataset k562"
            )
        
        logger.info(f"Loading K562 {self.split} data from {file_path}")
        
        # Load and filter data
        all_sequences, all_labels = self._load_and_filter_data(file_path)
        
        # Get or create splits using HashFrag (or fallback to chromosome-based)
        if self.use_hashfrag:
            try:
                splits = self._get_or_create_hashfrag_splits(
                    all_sequences,
                    all_labels,
                    data_dir
                )
            except RuntimeError as e:
                if self.use_chromosome_fallback:
                    logger.warning(f"HashFrag failed: {e}")
                    logger.warning("Falling back to chromosome-based splits")
                    splits = self._create_chromosome_splits(all_sequences, all_labels)
                else:
                    raise
        else:
            if self.use_chromosome_fallback:
                splits = self._create_chromosome_splits(all_sequences, all_labels)
            else:
                raise ValueError(
                    "use_hashfrag=False requires use_chromosome_fallback=True. "
                    "Please enable fallback if you want chromosome-based splits."
                )
        
        # Extract requested split
        self.sequences, self.labels, self.indices = splits[self.split]
        
        # Standardize sequences to 200bp
        self.sequences = self._standardize_to_200bp(self.sequences)
        
        # Apply subset size if specified (for downsampling experiments)
        # Use truly random sampling without replacement
        if self.subset_size is not None and self.subset_size < len(self.sequences):
            # No seed set - use current random state for maximum randomness
            indices = np.random.choice(len(self.sequences), size=self.subset_size, replace=False)
            self.sequences = self.sequences[indices]
            self.labels = self.labels[indices]
            logger.info(f"Downsampled to {self.subset_size:,} sequences (random sampling, no replacement)")
        
        self.sequence_length = self.SEQUENCE_LENGTH
        
        logger.info(f"Loaded {len(self.sequences)} sequences for {self.split} split")
        logger.info(f"Label range: [{np.min(self.labels):.3f}, {np.max(self.labels):.3f}]")
    
    def _load_and_filter_data(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and filter K562 data to reference alleles only.
        
        Returns:
            Tuple of (sequences, labels) for all filtered data
        """
        # Load data (tab-separated with header)
        try:
            df = pd.read_csv(file_path, sep='\t', dtype={'OL': str})
        except Exception as e:
            raise RuntimeError(f"Error loading K562 data from {file_path}: {e}")
        
        # Filter to reference alleles only (matching benchmark paper: 367K sequences)
        # Parse ID format: chr:pos:ref:alt:allele_type:wc
        id_parts = df['IDs'].str.split(':', expand=True)
        allele_type = id_parts[4]  # R=reference, A=alternate, empty=CRE/no variant
        ref_col = id_parts[2]
        alt_col = id_parts[3]
        
        # Keep reference alleles (R) and non-variant sequences (NA:NA)
        is_reference = allele_type == 'R'
        is_non_variant = (ref_col == 'NA') & (alt_col == 'NA')
        n_before = len(df)
        df = df[is_reference | is_non_variant].copy()
        n_after = len(df)
        
        logger.info(f"Filtered to {n_after:,} reference alleles (excluded {n_before - n_after:,} alternate alleles)")
        
        # Filter by sequence length (paper uses sequences >= 198bp for ~367K total)
        df['seq_len'] = df['sequence'].str.len()
        n_before_len = len(df)
        df = df[df['seq_len'] >= 198].copy()
        n_after_len = len(df)
        
        logger.info(f"Length filter (>= 198bp): {n_after_len:,} sequences (excluded {n_before_len - n_after_len:,} shorter sequences)")
        df = df.drop(columns=['seq_len'])
        
        # Extract sequences and labels
        sequences = df['sequence'].values
        labels = df['K562_log2FC'].values.astype(np.float32)
        
        return sequences, labels
    
    def _get_or_create_hashfrag_splits(
        self,
        all_sequences: np.ndarray,
        all_labels: np.ndarray,
        data_dir: Path
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get cached HashFrag splits or create new ones.
        
        Returns:
            Dict with 'train', 'pool', 'val', 'test' keys.
            Each value: (sequences, labels, indices)
        """
        # Set cache directory
        if self.hashfrag_cache_dir:
            cache_dir = Path(self.hashfrag_cache_dir)
        else:
            cache_dir = data_dir / 'hashfrag_splits'
        
        cache_files = {
            'train': cache_dir / 'train_indices.npy',
            'pool': cache_dir / 'pool_indices.npy',
            'val': cache_dir / 'val_indices.npy',
            'test': cache_dir / 'test_indices.npy'
        }
        
        # Check if cache exists
        if all(f.exists() for f in cache_files.values()):
            logger.info(f"Loading cached HashFrag splits from {cache_dir}")
            return self._load_cached_splits(all_sequences, all_labels, cache_files)
        
        # Create new splits
        logger.info("=" * 70)
        logger.info("Creating new HashFrag splits")
        logger.info("This will take several hours for the full K562 dataset...")
        logger.info("=" * 70)
        
        return self._create_new_hashfrag_splits(
            all_sequences,
            all_labels,
            cache_dir
        )
    
    def _load_cached_splits(
        self,
        all_sequences: np.ndarray,
        all_labels: np.ndarray,
        cache_files: Dict[str, Path]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load splits from cached indices."""
        splits = {}
        
        for split_name, cache_file in cache_files.items():
            indices = np.load(cache_file)
            sequences = all_sequences[indices]
            labels = all_labels[indices]
            splits[split_name] = (sequences, labels, indices)
            logger.info(f"  {split_name}: {len(indices):,} sequences")
        
        return splits
    
    def _create_new_hashfrag_splits(
        self,
        all_sequences: np.ndarray,
        all_labels: np.ndarray,
        cache_dir: Path
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Create new HashFrag splits and cache them."""
        # Create splitter
        splitter = HashFragSplitter(
            work_dir=str(cache_dir / "hashfrag_work"),
            threshold=self.hashfrag_threshold
        )
        
        # Create 80/10/10 splits
        # Note: HashFrag will create train/val/test
        # We'll further split train into train (100K) + pool (rest)
        raw_splits = splitter.create_splits_from_dataset(
            sequences=all_sequences,
            labels=all_labels,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            skip_revcomp=False  # Consider reverse complements as different
        )
        
        # Get the 80% training data
        train_pool_seqs, train_pool_labels, train_pool_indices = raw_splits['train']
        
        # Split into train and pool
        # For baseline experiments, we use the full train pool (no 100K cap)
        # The train pool is the full 80% of data, pool is not used for baseline
        # We'll use all of train_pool_indices as the train split
        n_train = len(train_pool_indices)  # Use full train pool, no cap
        
        # Randomly shuffle and split (no seed - use current random state for maximum randomness)
        shuffle_idx = np.random.permutation(len(train_pool_indices))
        
        train_shuffle = shuffle_idx[:n_train]
        pool_shuffle = shuffle_idx[n_train:]
        
        # Create final splits
        splits = {
            'train': (
                train_pool_seqs[train_shuffle],
                train_pool_labels[train_shuffle],
                train_pool_indices[train_shuffle]
            ),
            'pool': (
                train_pool_seqs[pool_shuffle],
                train_pool_labels[pool_shuffle],
                train_pool_indices[pool_shuffle]
            ),
            'val': raw_splits['val'],
            'test': raw_splits['test']
        }
        
        # Cache indices
        cache_dir.mkdir(parents=True, exist_ok=True)
        for split_name, (_, _, indices) in splits.items():
            cache_file = cache_dir / f'{split_name}_indices.npy'
            np.save(cache_file, indices)
            logger.info(f"Cached {split_name} indices: {len(indices):,} sequences")
        
        logger.info("✓ HashFrag splits created and cached!")
        return splits
    
    def _create_chromosome_splits(
        self,
        all_sequences: np.ndarray,
        all_labels: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Fallback: Create chromosome-based splits.
        
        This is less accurate than HashFrag but provides a reasonable approximation.
        Test: chr 7, 13 (~10%)
        Val: chr 19, 21, X (~10%)
        Train+Pool: remaining chromosomes (~80%)
        """
        logger.warning(
            "Using chromosome-based splits as fallback. "
            "This may result in some data leakage from homologous sequences. "
            "For publication-quality results, use HashFrag splits."
        )
        
        # This method would need chromosome information
        # For now, raise an error as we need to implement this properly
        raise NotImplementedError(
            "Chromosome-based fallback not yet implemented. "
            "Please install BLAST+ and use HashFrag for proper splits."
        )
    
    def _standardize_to_200bp(self, sequences: np.ndarray) -> np.ndarray:
        """
        Standardize sequences to 200bp.
        
        Sequences shorter than 200bp are padded equally on both ends with Ns.
        Sequences longer than 200bp are truncated (center-aligned).
        """
        processed = []
        
        for seq in sequences:
            curr_len = len(seq)
            
            if curr_len < self.SEQUENCE_LENGTH:
                # Pad equally on both ends with Ns
                pad_needed = self.SEQUENCE_LENGTH - curr_len
                left_pad = pad_needed // 2
                right_pad = pad_needed - left_pad
                padded = 'N' * left_pad + seq + 'N' * right_pad
                processed.append(padded)
                
            elif curr_len > self.SEQUENCE_LENGTH:
                # Truncate to target length (center-aligned)
                start = (curr_len - self.SEQUENCE_LENGTH) // 2
                processed.append(seq[start:start + self.SEQUENCE_LENGTH])
                
            else:
                processed.append(seq)
        
        # Verify all sequences are exactly 200bp
        for i, seq in enumerate(processed):
            if len(seq) != self.SEQUENCE_LENGTH:
                raise ValueError(f"Sequence {i} length mismatch: {len(seq)} != {self.SEQUENCE_LENGTH}")
        
        return np.array(processed)
    
    def encode_sequence(self, sequence: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Encode a K562 sequence with 5 channels.
        
        Args:
            sequence: DNA sequence string (200bp)
            metadata: Optional metadata dict (not used)
            
        Returns:
            Encoded sequence of shape (5, 200)
            Channels:
            - 0-3: one-hot encoded ACGT
            - 4: reverse complement flag (0 for forward, 1 for reverse)
        """
        # Get one-hot encoding (4 channels)
        encoded = one_hot_encode(sequence, add_singleton_channel=False)  # Shape: (4, 200)
        
        # Add reverse complement channel (always 0 for forward strand during training)
        rc_channel = np.zeros((1, len(sequence)), dtype=np.float32)
        
        # Concatenate: (4, 200) + (1, 200) = (5, 200)
        encoded = np.concatenate([encoded, rc_channel], axis=0)
        
        return encoded
    
    def get_num_channels(self) -> int:
        """Return number of input channels (5 for K562)."""
        return self.NUM_CHANNELS
