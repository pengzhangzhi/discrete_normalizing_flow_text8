import torch
import torch.nn as nn

from tarflow_single_system import TarFlow


class TextEncoder(nn.Module):
    """BERT-style bidirectional transformer encoder for text."""
    
    def __init__(
        self,
        vocab_size: int = 27,
        seq_len: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(vocab_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(seq_len, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: One-hot encoded input (B, seq_len, vocab_size)
        Returns:
            Continuous embeddings (B, seq_len, output_dim)
        """
        x = self.input_proj(x) + self.pos_embed
        x = self.transformer(x)
        x = self.output_proj(x)
        return x


class TarFlowModel(nn.Module):
    """Combined encoder + TarFlow normalizing flow model."""
    
    def __init__(
        self,
        # Encoder params
        vocab_size: int = 27,
        seq_len: int = 256,
        encoder_hidden: int = 256,
        encoder_output: int = 64,
        encoder_layers: int = 2,
        encoder_heads: int = 8,
        # TarFlow params
        flow_hidden: int = 768,
        flow_blocks: int = 12,
        flow_layers_per_block: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = TextEncoder(
            vocab_size=vocab_size,
            seq_len=seq_len,
            hidden_dim=encoder_hidden,
            output_dim=encoder_output,
            n_layers=encoder_layers,
            n_heads=encoder_heads,
            dropout=dropout,
        )
        
        # TarFlow expects: in_channels, img_size, patch_size, channels, num_blocks, layers_per_block
        # For text: num_patches = img_size // patch_size // in_channels = seq_len
        # So: img_size = seq_len * patch_size * in_channels
        self.flow = TarFlow(
            in_channels=encoder_output,
            img_size=seq_len * encoder_output,  # seq_len * in_channels (patch_size=1)
            patch_size=1,
            channels=flow_hidden,
            num_blocks=flow_blocks,
            layers_per_block=flow_layers_per_block,
            nvp=True,
            num_classes=0,
            dropout=dropout,
        )
        
        self.seq_len = seq_len
        self.encoder_output = encoder_output
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: One-hot encoded input (B, seq_len, vocab_size)
        Returns:
            z: Latent representation (B, seq_len, encoder_output)
            logdet: Log determinant of Jacobian (B,)
        """
        h = self.encoder(x)  # (B, seq_len, encoder_output)
        z, logdet = self.flow(h)  # z: (B, seq_len, encoder_output)
        return z, logdet
    
    def reverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reverse the flow to generate embeddings from latent.
        
        Args:
            z: Latent samples (B, seq_len, encoder_output)
        Returns:
            h: Continuous embeddings (B, seq_len, encoder_output)
        """
        return self.flow.reverse(z)
