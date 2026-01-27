import torch
import torch.nn as nn
import torch.nn.functional as F

from tarflow_single_system import TarFlow


class Attention(nn.Module):
    """Multi-head self-attention with SDPA."""
    
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
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.proj(x.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    """Transformer block with pre-norm."""
    
    def __init__(self, dim: int, n_heads: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class TextEncoder(nn.Module):
    """BERT-style bidirectional transformer encoder."""
    
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(vocab_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        self.blocks = nn.ModuleList([
            Block(hidden_dim, n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(hidden_dim)
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
        """x: (B, seq_len, vocab_size) one-hot -> (B, seq_len, hidden_dim)"""
        x = self.input_proj(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.ln_out(x)


class TarFlowModel(nn.Module):
    """Encoder + TarFlow normalizing flow with shared hidden dimension."""
    
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden_dim: int,
        encoder_layers: int,
        encoder_heads: int,
        flow_blocks: int,
        flow_layers_per_block: int,
        dropout: float = 0.0,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.encoder = TextEncoder(
            vocab_size=vocab_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            n_layers=encoder_layers,
            n_heads=encoder_heads,
            dropout=dropout,
        )
        self.flow = TarFlow(
            in_channels=hidden_dim,
            img_size=seq_len * hidden_dim,
            patch_size=1,
            channels=hidden_dim,
            num_blocks=flow_blocks,
            layers_per_block=flow_layers_per_block,
            nvp=True,
            num_classes=0,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, seq_len, vocab_size) -> z: (B, seq_len, hidden_dim), logdet: (B,)"""
        u = self.encoder(x)
        if self.training and self.noise_std > 0:
            u = u + self.noise_std * torch.randn_like(u)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            z, logdet = self.flow(u.float())
        return z, logdet
    
    def reverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reverse the flow: latent -> embeddings."""
        with torch.amp.autocast(device_type="cuda", enabled=False):
            return self.flow.reverse(z.float())
