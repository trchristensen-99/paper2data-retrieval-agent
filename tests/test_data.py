"""
Unit tests for data loading and preprocessing.
"""

import pytest
import numpy as np
from albench.data.utils import (
    one_hot_encode,
    reverse_complement,
    reverse_complement_one_hot,
    validate_sequence,
    pad_sequence,
    batch_encode_sequences
)


class TestOneHotEncoding:
    """Test one-hot encoding functions."""
    
    def test_basic_encoding(self):
        """Test basic one-hot encoding."""
        seq = "ACGT"
        encoded = one_hot_encode(seq)
        
        assert encoded.shape == (4, 4)
        assert encoded.dtype == np.float32
        
        # Check each nucleotide
        assert np.allclose(encoded[:, 0], [1, 0, 0, 0])  # A
        assert np.allclose(encoded[:, 1], [0, 1, 0, 0])  # C
        assert np.allclose(encoded[:, 2], [0, 0, 1, 0])  # G
        assert np.allclose(encoded[:, 3], [0, 0, 0, 1])  # T
    
    def test_singleton_channel(self):
        """Test encoding with singleton channel."""
        seq = "ACGT"
        
        # Non-singleton
        encoded = one_hot_encode(seq, add_singleton_channel=True, is_singleton=False)
        assert encoded.shape == (5, 4)
        assert np.allclose(encoded[4, :], [0, 0, 0, 0])
        
        # Singleton
        encoded = one_hot_encode(seq, add_singleton_channel=True, is_singleton=True)
        assert encoded.shape == (5, 4)
        assert np.allclose(encoded[4, :], [1, 1, 1, 1])
    
    def test_unknown_nucleotides(self):
        """Test encoding with unknown nucleotides (N)."""
        seq = "ACNGT"
        encoded = one_hot_encode(seq)
        
        # N should be all zeros
        assert np.allclose(encoded[:, 2], [0, 0, 0, 0])
    
    def test_case_insensitive(self):
        """Test that encoding is case-insensitive."""
        seq_upper = "ACGT"
        seq_lower = "acgt"
        
        encoded_upper = one_hot_encode(seq_upper)
        encoded_lower = one_hot_encode(seq_lower)
        
        assert np.allclose(encoded_upper, encoded_lower)


class TestReverseComplement:
    """Test reverse complement functions."""
    
    def test_reverse_complement_string(self):
        """Test reverse complement of DNA string."""
        assert reverse_complement("ACGT") == "ACGT"
        assert reverse_complement("AAAA") == "TTTT"
        assert reverse_complement("GCTA") == "TAGC"
        assert reverse_complement("ACGTN") == "NACGT"
    
    def test_reverse_complement_onehot(self):
        """Test reverse complement of one-hot encoded sequence."""
        seq = "ACGT"
        encoded = one_hot_encode(seq)
        rc_encoded = reverse_complement_one_hot(encoded)
        
        # Should match one-hot of reverse complement string
        rc_seq = reverse_complement(seq)
        expected = one_hot_encode(rc_seq)
        
        assert np.allclose(rc_encoded, expected)
    
    def test_reverse_complement_with_metadata(self):
        """Test that metadata channels are preserved."""
        seq = "ACGT"
        encoded = one_hot_encode(seq, add_singleton_channel=True, is_singleton=True)
        rc_encoded = reverse_complement_one_hot(encoded)
        
        # Singleton channel should be preserved
        assert np.allclose(rc_encoded[4, :], [1, 1, 1, 1])


class TestSequenceValidation:
    """Test sequence validation functions."""
    
    def test_valid_sequences(self):
        """Test validation of valid sequences."""
        assert validate_sequence("ACGT") == True
        assert validate_sequence("acgt") == True
        assert validate_sequence("ACGTN") == True
        assert validate_sequence("ACGTACGT") == True
    
    def test_invalid_sequences(self):
        """Test validation of invalid sequences."""
        assert validate_sequence("ACGTU") == False
        assert validate_sequence("ACG123") == False
        assert validate_sequence("ACG-T") == False


class TestPadding:
    """Test sequence padding functions."""
    
    def test_pad_right(self):
        """Test right padding."""
        seq = "ACGT"
        padded = pad_sequence(seq, 10, pad_char='N', mode='right')
        assert padded == "ACGTNNNNNN"
        assert len(padded) == 10
    
    def test_pad_left(self):
        """Test left padding."""
        seq = "ACGT"
        padded = pad_sequence(seq, 10, pad_char='N', mode='left')
        assert padded == "NNNNNNACGT"
        assert len(padded) == 10
    
    def test_pad_both(self):
        """Test centered padding."""
        seq = "ACGT"
        padded = pad_sequence(seq, 10, pad_char='N', mode='both')
        assert len(padded) == 10
        assert "ACGT" in padded
    
    def test_pad_exact_length(self):
        """Test padding when sequence is already correct length."""
        seq = "ACGT"
        padded = pad_sequence(seq, 4, pad_char='N', mode='both')
        assert padded == seq
    
    def test_pad_too_long(self):
        """Test error when sequence is longer than target."""
        seq = "ACGTACGT"
        with pytest.raises(ValueError):
            pad_sequence(seq, 4, pad_char='N', mode='both')


class TestBatchEncoding:
    """Test batch encoding functions."""
    
    def test_batch_encode_basic(self):
        """Test batch encoding of multiple sequences."""
        sequences = ["ACGT", "GCTA", "TGCA"]
        encoded = batch_encode_sequences(sequences)
        
        assert encoded.shape == (3, 4, 4)
        
        # Check individual sequences
        for i, seq in enumerate(sequences):
            expected = one_hot_encode(seq)
            assert np.allclose(encoded[i], expected)
    
    def test_batch_encode_with_singleton(self):
        """Test batch encoding with singleton flags."""
        sequences = ["ACGT", "GCTA"]
        is_singleton_flags = [True, False]
        
        encoded = batch_encode_sequences(
            sequences,
            add_singleton_channel=True,
            is_singleton_flags=is_singleton_flags
        )
        
        assert encoded.shape == (2, 5, 4)
        assert np.allclose(encoded[0, 4, :], [1, 1, 1, 1])
        assert np.allclose(encoded[1, 4, :], [0, 0, 0, 0])
    
    def test_batch_encode_different_lengths_error(self):
        """Test error when sequences have different lengths."""
        sequences = ["ACGT", "ACGTACGT"]
        
        with pytest.raises(ValueError):
            batch_encode_sequences(sequences)
    
    def test_batch_encode_empty_error(self):
        """Test error with empty sequence list."""
        with pytest.raises(ValueError):
            batch_encode_sequences([])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
