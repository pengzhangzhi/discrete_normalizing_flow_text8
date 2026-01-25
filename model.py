import torch
import torch.nn as nn
import torch.nn.functional as F

from tarflow_single_system import TarFlow


class Attention(nn.Module):
    """Multi-head self-attention with SDPA (auto Flash/mem-efficient kernels)."""
    
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # 3 x (B, H, T, D)
        
        # SDPA: auto-selects Flash/mem-efficient/math backend
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # bidirectional
        )
        
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    """Transformer block with pre-norm."""
    
    def __init__(self, dim: int, n_heads: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, expansion, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class TextEncoder(nn.Module):
    """BERT-style bidirectional transformer encoder with SDPA."""
    
    def __init__(
        self,
        vocab_size: int = 27,
        seq_len: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 8,
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(vocab_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        
        self.blocks = nn.ModuleList([
            Block(hidden_dim, n_heads, expansion, dropout) 
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: One-hot encoded input (B, seq_len, vocab_size)
        Returns:
            Continuous embeddings (B, seq_len, output_dim)
        """
        x = self.input_proj(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
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
        
        # TarFlow: num_patches = img_size // patch_size // in_channels
        # For seq_len patches: img_size = seq_len * in_channels (with patch_size=1)
        self.flow = TarFlow(
            in_channels=encoder_output,
            img_size=seq_len * encoder_output,
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
        h = self.encoder(x)
        z, logdet = self.flow(h)
        return z, logdet
    
    def reverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reverse the flow: latent -> embeddings."""
        return self.flow.reverse(z)
