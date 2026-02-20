"""
HashFrag wrapper for creating homology-aware data splits.

HashFrag identifies homologous sequences using BLAST and Smith-Waterman alignment,
then creates orthogonal train/validation/test splits where no homologous sequences
span multiple splits. This prevents data leakage from sequence similarity.

Example usage:
    from data.hashfrag_splits import HashFragSplitter
    
    splitter = HashFragSplitter(
        work_dir="./data/hashfrag_work/k562",
        threshold=60
    )
    
    splits = splitter.create_splits_from_dataset(
        sequences=seq_array,
        labels=expr_array,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    train_seqs, train_labels, train_indices = splits['train']
"""

import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)


class HashFragSplitter:
    """
    Python wrapper for HashFrag tool to create homology-aware data splits.
    
    HashFrag prevents data leakage by ensuring that homologous (similar) sequences
    don't end up in different splits (e.g., training vs testing).
    
    How it works:
    1. Runs BLAST to find candidate homologous pairs
    2. Computes exact Smith-Waterman alignment scores
    3. Groups sequences into homology clusters (graph-based)
    4. Distributes entire clusters across train/val/test splits
    
    Args:
        hashfrag_path: Path to hashFrag executable (default: searches PATH)
        work_dir: Directory for intermediate files (default: ./data/hashfrag_work)
        threshold: Smith-Waterman score threshold for defining homology (default: 60)
        scoring_params: Dict with 'reward', 'penalty', 'gapopen', 'gapextend'
                       (default: match=1, mismatch=-1, gapopen=2, gapextend=1)
    """
    
    def __init__(
        self,
        hashfrag_path: str = "hashFrag",
        work_dir: str = "./data/hashfrag_work",
        threshold: int = 60,
        scoring_params: Optional[Dict[str, int]] = None
    ):
        self.hashfrag_path = hashfrag_path
        self.work_dir = Path(work_dir)
        self.threshold = threshold
        
        # Default scoring parameters match pilot study
        # Note: penalty must be negative for BLAST, but HashFrag expects positive
        # and converts it internally
        self.scoring_params = scoring_params or {
            'reward': 1,      # match score (positive)
            'penalty': -1,    # mismatch penalty (negative for BLAST)
            'gapopen': 2,     # gap opening penalty (positive, HashFrag converts)
            'gapextend': 1    # gap extension penalty (positive, HashFrag converts)
        }
        
        # Create work directory
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Check dependencies
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check that BLAST+ and HashFrag are installed."""
        # Check BLAST+
        if not shutil.which('blastn'):
            raise RuntimeError(
                "BLAST+ not found! Please install it:\n"
                "  Ubuntu/Debian: sudo apt-get install ncbi-blast+\n"
                "  macOS: brew install blast\n"
                "  HPC: module load blast-plus\n"
                "Or download from: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/"
            )
        
        # Check HashFrag
        if not shutil.which(self.hashfrag_path):
            raise RuntimeError(
                f"HashFrag not found at '{self.hashfrag_path}'!\n"
                "Please run: ./scripts/setup_hashfrag.sh\n"
                "Then add to PATH: export PATH=\"$PATH:$(pwd)/external/hashFrag/src\""
            )
        
        logger.info("✓ BLAST+ and HashFrag dependencies found")
    
    def sequences_to_fasta(
        self,
        sequences: np.ndarray,
        output_path: Path,
        ids: Optional[List[str]] = None
    ) -> Path:
        """
        Convert numpy array of sequences to FASTA format.
        
        Args:
            sequences: Array of DNA sequence strings
            output_path: Where to write FASTA file
            ids: Optional sequence IDs (default: "seq_0", "seq_1", ...)
        
        Returns:
            Path to created FASTA file
        
        Example:
            >>> seqs = np.array(['ATCG', 'GCTA', 'TACG'])
            >>> splitter.sequences_to_fasta(seqs, Path('test.fa'))
            PosixPath('test.fa')
        """
        logger.info(f"Writing {len(sequences)} sequences to FASTA: {output_path}")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"seq_{i}" for i in range(len(sequences))]
        
        if len(ids) != len(sequences):
            raise ValueError(f"Number of IDs ({len(ids)}) doesn't match sequences ({len(sequences)})")
        
        # Write FASTA format
        with open(output_path, 'w') as f:
            for seq_id, sequence in zip(ids, sequences):
                f.write(f">{seq_id}\n")
                f.write(f"{sequence}\n")
        
        logger.info(f"✓ Wrote FASTA file: {output_path}")
        return output_path
    
    def run_hashfrag(
        self,
        fasta_path: Path,
        output_name: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        skip_revcomp: bool = False,
        threads: int = 16
    ) -> Path:
        """
        Run HashFrag to create orthogonal splits.
        
        This is the core HashFrag execution step. It will:
        1. Run BLAST to find candidate homologous sequences
        2. Compute exact Smith-Waterman scores for candidates
        3. Cluster sequences by homology
        4. Split clusters across train/val/test
        
        WARNING: This can take several hours for large datasets!
        
        Args:
            fasta_path: Input FASTA file with all sequences
            output_name: Name for output directory
            train_ratio: Fraction for training (default: 0.8)
            val_ratio: Fraction for validation (default: 0.1)
            test_ratio: Fraction for testing (default: 0.1)
            skip_revcomp: Skip reverse complement consideration (default: False)
        
        Returns:
            Path to output directory containing split files
        
        Raises:
            RuntimeError: If HashFrag execution fails
        """
        if not fasta_path.exists():
            raise FileNotFoundError(f"Input FASTA not found: {fasta_path}")
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_ratio} "
                f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
            )
        
        output_dir = self.work_dir / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info("Running HashFrag - this will take a while...")
        logger.info("=" * 70)
        logger.info(f"Input: {fasta_path}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Threshold: {self.threshold}")
        logger.info(f"Split ratio: {train_ratio:.1%}/{val_ratio:.1%}/{test_ratio:.1%}")
        logger.info("=" * 70)
        
        # Build HashFrag command
        # Note: HashFrag only does 2-way splits (train/test)
        # For hierarchical 3-way splits, we call this twice:
        #   1st call: (train+val) vs test → use train_ratio for (train+val), test_ratio for test
        #   2nd call: train vs val → use train_ratio for train, val_ratio for val
        cmd = [
            self.hashfrag_path,
            "create_orthogonal_splits",
            "-f", str(fasta_path),
            "-t", str(self.threshold),
            "-r", str(self.scoring_params['reward']),
            "-p", str(self.scoring_params['penalty']),
            "-g", str(self.scoring_params['gapopen']),
            "-x", str(self.scoring_params['gapextend']),
            "-T", str(threads),  # Number of CPU threads for BLAST
            "-o", str(output_dir),
            "--p-train", str(train_ratio),  # Proportion for "train" split
            "--p-test", str(val_ratio + test_ratio),  # Proportion for "test" split
            "--force"  # Force recompute BLAST even if cached
        ]
        
        if skip_revcomp:
            cmd.append("--skip-revcomp")
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Execute HashFrag
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Log output
            if result.stdout:
                logger.info("HashFrag output:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")
            
            logger.info("✓ HashFrag completed successfully!")
            return output_dir
            
        except subprocess.CalledProcessError as e:
            error_msg = f"HashFrag failed with exit code {e.returncode}\n"
            if e.stderr:
                error_msg += f"Error output:\n{e.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def parse_tsv_splits(self, tsv_file_path: Path) -> Dict[str, np.ndarray]:
        """
        Parse HashFrag TSV output file with split assignments.
        
        HashFrag output format: TSV with columns 'id' and 'split'
        
        Args:
            tsv_file_path: Path to TSV split file
        
        Returns:
            Dictionary mapping split names to arrays of sequence IDs
        
        Example:
            >>> splits = splitter.parse_tsv_splits(Path('split_001.tsv'))
            >>> print(splits['train'][:3])
            ['seq_0', 'seq_15', 'seq_23']
        """
        if not tsv_file_path.exists():
            raise FileNotFoundError(f"Split file not found: {tsv_file_path}")
        
        import pandas as pd
        df = pd.read_csv(tsv_file_path, sep='\t')
        
        # Filter out reverse complement sequences (_Reversed suffix)
        df = df[~df['id'].str.endswith('_Reversed')]
        
        # Group by split
        splits = {}
        for split_name in df['split'].unique():
            ids = df[df['split'] == split_name]['id'].values
            splits[split_name] = ids
            logger.info(f"Parsed {len(ids):,} IDs for '{split_name}' split")
        
        return splits
    
    def create_splits_from_dataset(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        sequence_ids: Optional[List[str]] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        skip_revcomp: bool = False
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Complete end-to-end pipeline to create homology-aware splits.
        
        This is the main method you'll use. It handles:
        1. Writing sequences to FASTA
        2. Running HashFrag
        3. Parsing output
        4. Mapping IDs back to original indices
        5. Extracting sequences/labels for each split
        
        Args:
            sequences: Array of DNA sequences (shape: [n_sequences])
            labels: Array of labels (shape: [n_sequences])
            sequence_ids: Optional sequence IDs (default: auto-generated)
            train_ratio: Fraction for training (default: 0.8)
            val_ratio: Fraction for validation (default: 0.1)
            test_ratio: Fraction for testing (default: 0.1)
            skip_revcomp: Skip reverse complement (default: False)
        
        Returns:
            Dictionary with 'train', 'val', 'test' keys.
            Each value is a tuple: (sequences, labels, indices)
            
            - sequences: np.ndarray of DNA sequences for this split
            - labels: np.ndarray of labels for this split
            - indices: np.ndarray of indices into original arrays
        
        Example:
            >>> splits = splitter.create_splits_from_dataset(seqs, labels)
            >>> train_seqs, train_labels, train_idx = splits['train']
            >>> print(f"Training on {len(train_seqs)} sequences")
        """
        logger.info(f"Creating HashFrag splits for {len(sequences)} sequences...")
        
        # Validate inputs
        if len(sequences) != len(labels):
            raise ValueError(
                f"Sequences ({len(sequences)}) and labels ({len(labels)}) "
                "must have same length"
            )
        
        # Generate sequence IDs if not provided
        if sequence_ids is None:
            sequence_ids = [f"seq_{i}" for i in range(len(sequences))]
        
        # Create ID -> index mapping (needed for second HashFrag run)
        id_to_idx = {seq_id: idx for idx, seq_id in enumerate(sequence_ids)}
        
        # Step 1: Write sequences to FASTA
        fasta_path = self.work_dir / "input_sequences.fa"
        self.sequences_to_fasta(sequences, fasta_path, sequence_ids)
        
        # Step 2: Run HashFrag
        # First split: (train+val) vs test
        # We want 90% (train+val) vs 10% (test)
        # So pass train_ratio=0.9, test_ratio=0.1 for the first split
        train_val_ratio = train_ratio + val_ratio  # 0.9 for 80:10:10
        test_ratio_first = test_ratio  # 0.1 for 80:10:10
        
        output_dir = self.run_hashfrag(
            fasta_path,
            output_name="hashfrag_output",
            train_ratio=train_val_ratio,  # 0.9 for first split
            val_ratio=0.0,  # Not used in first split
            test_ratio=test_ratio_first,  # 0.1 for first split
            skip_revcomp=skip_revcomp
        )
        
        # Step 3: Parse first-level split (train+val vs test)
        # HashFrag creates a single TSV file with split assignments
        tsv_files = list(output_dir.glob('hashFrag.*.split_*.tsv'))
        if not tsv_files:
            raise FileNotFoundError(
                f"HashFrag output TSV file missing in: {output_dir}\n"
                "HashFrag may have failed silently"
            )
        first_split_tsv = tsv_files[0]  # Should only be one
        logger.info(f"Parsing first split from: {first_split_tsv.name}")
        
        first_splits = self.parse_tsv_splits(first_split_tsv)
        train_val_ids = first_splits['train']  # Contains train+val for first split
        test_ids = first_splits['test']
        
        logger.info("First-level split complete (train+val vs test)")
        logger.info(f"  Train+Val: {len(train_val_ids):,} sequences")
        logger.info(f"  Test: {len(test_ids):,} sequences")
        
        # Step 4: Run HashFrag AGAIN on train+val to split into train and val
        # This ensures orthogonality between train and val as well
        logger.info("Running second HashFrag to split train+val orthogonally...")
        
        # Write train+val sequences to new FASTA
        train_val_fasta = self.work_dir / "train_val_sequences.fa"
        train_val_indices = np.array([id_to_idx[seq_id] for seq_id in train_val_ids])
        train_val_sequences = sequences[train_val_indices]
        
        self.sequences_to_fasta(train_val_sequences, train_val_fasta, list(train_val_ids))
        
        # Calculate val fraction within train+val
        val_fraction_of_trainval = val_ratio / (train_ratio + val_ratio)
        
        # Run second HashFrag
        second_output_dir = self.run_hashfrag(
            train_val_fasta,
            output_name="hashfrag_output_second",
            train_ratio=1.0 - val_fraction_of_trainval,  # train within train+val
            val_ratio=val_fraction_of_trainval,  # This becomes "test" in HashFrag's 2-way split
            test_ratio=0.0,  # Not used in second split
            skip_revcomp=skip_revcomp
        )
        
        # Parse second-level split
        second_tsv_files = list(second_output_dir.glob('hashFrag.*.split_*.tsv'))
        if not second_tsv_files:
            raise FileNotFoundError(
                f"Second HashFrag split TSV missing in: {second_output_dir}"
            )
        second_split_tsv = second_tsv_files[0]
        logger.info(f"Parsing second split from: {second_split_tsv.name}")
        
        second_splits = self.parse_tsv_splits(second_split_tsv)
        train_ids_final = second_splits['train']  # Final train set
        val_ids_final = second_splits['test']  # What HashFrag calls "test" is our "val"
        
        logger.info("Second-level split complete (train vs val)")
        logger.info(f"  Train: {len(train_ids_final):,} sequences")
        logger.info(f"  Val: {len(val_ids_final):,} sequences")
        
        split_ids = {
            'train': train_ids_final,
            'val': val_ids_final,
            'test': test_ids
        }
        
        # Step 4: Map IDs back to indices
        # Create ID -> index mapping
        id_to_idx = {seq_id: idx for idx, seq_id in enumerate(sequence_ids)}
        
        # Convert IDs to indices for each split
        splits = {}
        for split_name, ids in split_ids.items():
            # Map IDs to indices
            indices = np.array([id_to_idx[seq_id] for seq_id in ids])
            
            # Extract sequences and labels for this split
            split_sequences = sequences[indices]
            split_labels = labels[indices]
            
            splits[split_name] = (split_sequences, split_labels, indices)
            
            logger.info(
                f"{split_name.capitalize()} split: {len(indices)} sequences "
                f"({len(indices)/len(sequences):.1%})"
            )
        
        # Verify all sequences are assigned
        total_assigned = sum(len(splits[s][2]) for s in ['train', 'val', 'test'])
        if total_assigned != len(sequences):
            logger.warning(
                f"Some sequences not assigned! "
                f"Total: {len(sequences)}, Assigned: {total_assigned}"
            )
        
        logger.info("✓ HashFrag splits created successfully!")
        return splits
