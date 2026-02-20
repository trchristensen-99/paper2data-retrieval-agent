"""
Utility functions for sequence encoding and manipulation.

This module provides functions for:
- One-hot encoding DNA sequences
- Computing reverse complements
- Sequence validation and normalization
"""

from typing import Dict, List, Union
import numpy as np


# Nucleotide mappings
NUCLEOTIDE_TO_INDEX: Dict[str, int] = {
    'A': 0, 'a': 0,
    'C': 1, 'c': 1,
    'G': 2, 'g': 2,
    'T': 3, 't': 3,
}

COMPLEMENT_MAP: Dict[str, str] = {
    'A': 'T', 'T': 'A',
    'C': 'G', 'G': 'C',
    'a': 't', 't': 'a',
    'c': 'g', 'g': 'c',
}


def one_hot_encode(
    sequence: str, 
    num_channels: int = 4,
    add_singleton_channel: bool = False,
    is_singleton: bool = False
) -> np.ndarray:
    """
    One-hot encode a DNA sequence.
    
    Args:
        sequence: DNA sequence string (ACGT)
        num_channels: Number of channels for encoding (4 for ACGT, 5 for ACGT+singleton)
        add_singleton_channel: Whether to add a singleton indicator channel
        is_singleton: Whether this sequence is a singleton (only if add_singleton_channel=True)
    
    Returns:
        One-hot encoded array of shape (num_channels, sequence_length)
        
    Examples:
        >>> seq = "ACGT"
        >>> encoded = one_hot_encode(seq)
        >>> encoded.shape
        (4, 4)
        >>> encoded[:, 0]  # 'A' is [1, 0, 0, 0]
        array([1., 0., 0., 0.])
    """
    sequence = sequence.upper()
    seq_length = len(sequence)
    
    # Initialize array
    if add_singleton_channel:
        encoded = np.zeros((5, seq_length), dtype=np.float32)
    else:
        encoded = np.zeros((4, seq_length), dtype=np.float32)
    
    # Encode nucleotides
    for i, nucleotide in enumerate(sequence):
        if nucleotide in NUCLEOTIDE_TO_INDEX:
            idx = NUCLEOTIDE_TO_INDEX[nucleotide]
            encoded[idx, i] = 1.0
        # Unknown nucleotides (N, etc.) remain as zeros
    
    # Add singleton channel if requested
    if add_singleton_channel:
        encoded[4, :] = 1.0 if is_singleton else 0.0
    
    return encoded


def reverse_complement(sequence: str) -> str:
    """
    Compute the reverse complement of a DNA sequence.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Reverse complement of the input sequence
        
    Examples:
        >>> reverse_complement("ACGT")
        'ACGT'
        >>> reverse_complement("AAAA")
        'TTTT'
        >>> reverse_complement("ACGTN")
        'NACGT'
    """
    complement = ''.join(COMPLEMENT_MAP.get(base, base) for base in sequence)
    return complement[::-1]


def reverse_complement_one_hot(encoded: np.ndarray) -> np.ndarray:
    """
    Compute reverse complement of a one-hot encoded sequence.
    
    Args:
        encoded: One-hot encoded array of shape (num_channels, seq_length)
                First 4 channels are ACGT, optional 5th channel is metadata
                
    Returns:
        Reverse complement of the encoded sequence
        
    Note:
        - Reverses along sequence dimension
        - Swaps A<->T (channels 0<->3) and C<->G (channels 1<->2)
        - Preserves metadata channels (5th channel and beyond)
    """
    # Reverse along sequence dimension
    rc_encoded = encoded[:, ::-1].copy()
    
    # Swap channels for complement: A<->T (0<->3), C<->G (1<->2)
    rc_encoded[[0, 1, 2, 3], :] = rc_encoded[[3, 2, 1, 0], :]
    
    return rc_encoded


def validate_sequence(sequence: str, allowed_chars: str = "ACGTN") -> bool:
    """
    Validate that a sequence contains only allowed characters.
    
    Args:
        sequence: DNA sequence string
        allowed_chars: String of allowed characters (case-insensitive)
        
    Returns:
        True if sequence is valid, False otherwise
    """
    allowed = set(allowed_chars.upper() + allowed_chars.lower())
    return all(char in allowed for char in sequence)


def pad_sequence(
    sequence: str, 
    target_length: int,
    pad_char: str = 'N',
    mode: str = 'both'
) -> str:
    """
    Pad a sequence to a target length.
    
    Args:
        sequence: DNA sequence to pad
        target_length: Desired final length
        pad_char: Character to use for padding
        mode: Padding mode - 'left', 'right', or 'both' (centered)
        
    Returns:
        Padded sequence
        
    Raises:
        ValueError: If sequence is longer than target_length
    """
    if len(sequence) >= target_length:
        if len(sequence) == target_length:
            return sequence
        raise ValueError(
            f"Sequence length {len(sequence)} exceeds target length {target_length}"
        )
    
    pad_needed = target_length - len(sequence)
    
    if mode == 'left':
        return pad_char * pad_needed + sequence
    elif mode == 'right':
        return sequence + pad_char * pad_needed
    elif mode == 'both':
        left_pad = pad_needed // 2
        right_pad = pad_needed - left_pad
        return pad_char * left_pad + sequence + pad_char * right_pad
    else:
        raise ValueError(f"Unknown padding mode: {mode}")


def batch_encode_sequences(
    sequences: List[str],
    add_singleton_channel: bool = False,
    is_singleton_flags: Union[List[bool], None] = None
) -> np.ndarray:
    """
    Batch encode multiple sequences into a single array.
    
    Args:
        sequences: List of DNA sequences (must all be same length)
        add_singleton_channel: Whether to add singleton indicator channel
        is_singleton_flags: List of boolean flags for each sequence (only if add_singleton_channel=True)
        
    Returns:
        Array of shape (num_sequences, num_channels, seq_length)
        
    Raises:
        ValueError: If sequences have different lengths
    """
    if not sequences:
        raise ValueError("Cannot encode empty sequence list")
    
    seq_lengths = [len(seq) for seq in sequences]
    if len(set(seq_lengths)) > 1:
        raise ValueError(
            f"All sequences must have same length, got lengths: {set(seq_lengths)}"
        )
    
    if add_singleton_channel and is_singleton_flags is None:
        is_singleton_flags = [False] * len(sequences)
    
    encoded_list = []
    for i, seq in enumerate(sequences):
        is_singleton = is_singleton_flags[i] if is_singleton_flags else False
        encoded = one_hot_encode(
            seq, 
            add_singleton_channel=add_singleton_channel,
            is_singleton=is_singleton
        )
        encoded_list.append(encoded)
    
    return np.stack(encoded_list, axis=0)
