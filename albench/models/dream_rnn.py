"""
DREAM-RNN model architecture for sequence-to-function prediction.

Based on the architecture from Prix Fixe (de Boer Lab) used in the
Random Promoter DREAM Challenge.

Architecture:
1. First layer: Dual CNNs with kernel sizes [9, 15] → concatenate
2. Core: Bi-LSTM → CNN
3. Final: Conv1D → global average pooling → linear output
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from .base import SequenceModel


class DREAMRNN(SequenceModel):
    """
    DREAM-RNN architecture for genomic sequence-to-function prediction.
    
    Based on Prix Fixe implementation from de Boer Lab.
    
    Architecture details:
    - First layer block: Two parallel 1D CNNs with kernel sizes (9, 15)
      to capture motifs of different lengths. Each has 160 channels (320 total after concat).
    - Core block: Bidirectional LSTM (320 units each direction = 640 total)
      followed by another dual CNN block (same as first layer).
    - Final block: 1D conv (256 filters) → global average pooling → linear layer.
    
    Task-specific final layers:
    - Yeast: 18-bin SoftMax classification → weighted average to expression (KL divergence loss)
    - K562: Direct regression output (MSE loss)
    
    The model uses dropout for regularization and supports sequences with
    different numbers of input channels (4 for ACGT, 5 for ACGT+singleton).
    
    Args:
        input_channels: Number of input channels (4 for yeast ACGT, 5 for K562 ACGT+singleton)
        sequence_length: Length of input sequences (80 for yeast, 230 for K562)
        output_dim: Output dimension (1 for K562 regression, 18 for yeast bins, or auto-detect from task_mode)
        hidden_dim: LSTM hidden dimension per direction (default: 320)
        cnn_filters: Number of filters PER CNN in dual blocks (default: 160, so 320 total after concat)
        dropout_cnn: Dropout rate after CNN layers (default: 0.2 in original, 0.1 for MC dropout)
        dropout_lstm: Dropout rate after LSTM (default: 0.5 in original, 0.1 for MC dropout)
        task_mode: 'yeast' for 18-bin SoftMax classification, 'k562' for direct regression (default: 'k562')
    """
    
    def __init__(
        self,
        input_channels: int,
        sequence_length: int,
        output_dim: Optional[int] = None,
        hidden_dim: int = 320,
        cnn_filters: int = 160,  # Original Prix Fixe: 160 per CNN → 320 total after concat
        dropout_cnn: float = 0.2,
        dropout_lstm: float = 0.5,
        task_mode: str = 'k562'  # 'yeast' or 'k562'
    ):
        # Auto-detect output_dim from task_mode if not specified
        if output_dim is None:
            if task_mode == 'yeast':
                output_dim = 18  # 18 bins for yeast
            else:
                output_dim = 1  # Direct regression for K562
        
        super().__init__(input_channels, sequence_length, output_dim)
        
        self.task_mode = task_mode
        
        self.hidden_dim = hidden_dim
        self.cnn_filters = cnn_filters
        self.dropout_cnn = dropout_cnn
        self.dropout_lstm = dropout_lstm
        
        # First layer block: Dual CNNs with different kernel sizes
        # Rationale: Motifs are typically 9-15bp long
        self.conv1_short = nn.Conv1d(
            in_channels=input_channels,
            out_channels=cnn_filters,
            kernel_size=9,
            padding='same'  # Keep sequence length
        )
        
        self.conv1_long = nn.Conv1d(
            in_channels=input_channels,
            out_channels=cnn_filters,
            kernel_size=15,
            padding='same'
        )
        
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_cnn)
        
        # After concatenating, we have 2*cnn_filters channels (320 total with default)
        concat_channels = 2 * cnn_filters
        
        # Core block: Bi-LSTM
        # Input: (batch, seq_len, features) - need to permute from (batch, channels, seq_len)
        # hidden_dim per direction, so total output is 2*hidden_dim
        self.lstm = nn.LSTM(
            input_size=concat_channels,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # Only one layer, so no dropout between layers
        )
        
        # CNN block after LSTM (dual CNNs like first layer block)
        lstm_output_size = 2 * hidden_dim  # Bidirectional = 640
        self.conv2_short = nn.Conv1d(
            in_channels=lstm_output_size,
            out_channels=cnn_filters,
            kernel_size=9,
            padding='same'
        )
        
        self.conv2_long = nn.Conv1d(
            in_channels=lstm_output_size,
            out_channels=cnn_filters,
            kernel_size=15,
            padding='same'
        )
        
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_lstm)
        
        # After concatenating second CNN block: 2*cnn_filters = 320
        conv2_concat_channels = 2 * cnn_filters
        
        # Final block: Conv1D → global average pooling → linear
        self.conv3 = nn.Conv1d(
            in_channels=conv2_concat_channels,
            out_channels=256,  # Prix Fixe uses 256 in final conv
            kernel_size=1,  # Point-wise convolution
            padding='same'
        )
        self.relu3 = nn.ReLU()
        
        # Global average pooling will be done manually
        # Linear layer to output
        # For yeast: output 18 logits (before SoftMax)
        # For K562: output 1 value (direct regression)
        self.fc = nn.Linear(256, output_dim)
        
        # For yeast mode: bin centers for weighted average conversion
        # Expression values range from 0-17, so bins are [0, 1, 2, ..., 17]
        # Bin centers are at integer values (0.0, 1.0, 2.0, ..., 17.0)
        if task_mode == 'yeast':
            # Register bin centers as buffer (not a parameter, but part of model state)
            bin_centers = torch.arange(18, dtype=torch.float32)  # [0.0, 1.0, ..., 17.0]
            self.register_buffer('bin_centers', bin_centers)
            # SoftMax for converting logits to probabilities
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.bin_centers = None
            self.softmax = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DREAM-RNN.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Input shape: (batch, channels, seq_len)
        batch_size = x.size(0)
        
        # First layer block: Dual CNNs
        conv1_short_out = self.conv1_short(x)  # (batch, cnn_filters, seq_len)
        conv1_long_out = self.conv1_long(x)    # (batch, cnn_filters, seq_len)
        
        # Concatenate along channel dimension
        conv1_out = torch.cat([conv1_short_out, conv1_long_out], dim=1)  # (batch, 2*cnn_filters, seq_len)
        conv1_out = self.relu1(conv1_out)
        conv1_out = self.dropout1(conv1_out)
        
        # LSTM expects input of shape (batch, seq_len, features)
        lstm_in = conv1_out.permute(0, 2, 1)  # (batch, seq_len, 2*cnn_filters)
        
        # Core block: Bi-LSTM
        lstm_out, _ = self.lstm(lstm_in)  # (batch, seq_len, 2*hidden_dim)
        
        # Convert back to (batch, channels, seq_len) for CNN
        lstm_out = lstm_out.permute(0, 2, 1)  # (batch, 2*hidden_dim, seq_len)
        
        # Dual CNN block after LSTM (same structure as first layer)
        conv2_short_out = self.conv2_short(lstm_out)  # (batch, cnn_filters, seq_len)
        conv2_long_out = self.conv2_long(lstm_out)    # (batch, cnn_filters, seq_len)
        
        # Concatenate along channel dimension
        conv2_out = torch.cat([conv2_short_out, conv2_long_out], dim=1)  # (batch, 2*cnn_filters, seq_len)
        conv2_out = self.relu2(conv2_out)
        conv2_out = self.dropout2(conv2_out)
        
        # Final block: Point-wise Conv1D
        conv3_out = self.conv3(conv2_out)  # (batch, 256, seq_len)
        conv3_out = self.relu3(conv3_out)
        
        # Global average pooling along sequence dimension
        pooled = torch.mean(conv3_out, dim=2)  # (batch, 256)
        
        # Linear layer to output
        logits = self.fc(pooled)  # (batch, output_dim)
        
        # Task-specific output processing
        if self.task_mode == 'yeast':
            # Yeast: Convert 18-bin logits to expression via SoftMax + weighted average
            # logits shape: (batch, 18)
            bin_probs = self.softmax(logits)  # (batch, 18) - probabilities for each bin
            # Weighted average: expression = Σ(bin_center_i × prob_i)
            output = torch.sum(bin_probs * self.bin_centers.unsqueeze(0), dim=1)  # (batch,)
        else:
            # K562: Direct regression output
            # If output_dim == 1, squeeze to (batch,)
            if self.output_dim == 1:
                output = logits.squeeze(1)
            else:
                output = logits
        
        return output
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get raw logits before SoftMax (for yeast) or final output (for K562).
        Useful for computing KL divergence loss for yeast.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length)
            
        Returns:
            Logits tensor of shape (batch_size, 18) for yeast or (batch_size, 1) for K562
        """
        # Forward pass up to logits
        batch_size = x.size(0)
        conv1_short_out = self.conv1_short(x)
        conv1_long_out = self.conv1_long(x)
        conv1_out = torch.cat([conv1_short_out, conv1_long_out], dim=1)
        conv1_out = self.relu1(conv1_out)
        conv1_out = self.dropout1(conv1_out)
        lstm_in = conv1_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = lstm_out.permute(0, 2, 1)
        conv2_short_out = self.conv2_short(lstm_out)
        conv2_long_out = self.conv2_long(lstm_out)
        conv2_out = torch.cat([conv2_short_out, conv2_long_out], dim=1)
        conv2_out = self.relu2(conv2_out)
        conv2_out = self.dropout2(conv2_out)
        conv3_out = self.conv3(conv2_out)
        conv3_out = self.relu3(conv3_out)
        pooled = torch.mean(conv3_out, dim=2)
        logits = self.fc(pooled)
        
        return logits
    
    def get_bin_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get bin probabilities for yeast mode (before weighted average).
        Only valid when task_mode == 'yeast'.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length)
            
        Returns:
            Bin probabilities tensor of shape (batch_size, 18)
        """
        if self.task_mode != 'yeast':
            raise ValueError("get_bin_probabilities() only available for yeast task_mode")
        
        logits = self.get_logits(x)
        return self.softmax(logits)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        info = super().get_model_info()
        info.update({
            "hidden_dim": self.hidden_dim,
            "cnn_filters": self.cnn_filters,
            "dropout_cnn": self.dropout_cnn,
            "dropout_lstm": self.dropout_lstm,
            "task_mode": self.task_mode,
        })
        if self.task_mode == 'yeast':
            info["bin_centers"] = self.bin_centers.cpu().numpy().tolist()
        return info


def create_dream_rnn(
    input_channels: int,
    sequence_length: int,
    task_mode: str = 'k562',
    **kwargs
) -> DREAMRNN:
    """
    Factory function to create a DREAM-RNN model with sensible defaults.
    
    Args:
        input_channels: Number of input channels (4 or 5)
        sequence_length: Length of input sequences
        task_mode: 'yeast' for 18-bin SoftMax classification, 'k562' for direct regression
        **kwargs: Additional arguments passed to DREAMRNN constructor
        
    Returns:
        Initialized DREAMRNN model
        
    Example:
        >>> model = create_dream_rnn(input_channels=5, sequence_length=230, task_mode='k562')
        >>> print(model.get_model_info())
    """
    model = DREAMRNN(
        input_channels=input_channels,
        sequence_length=sequence_length,
        task_mode=task_mode,
        **kwargs
    )
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    model.apply(init_weights)
    
    return model
