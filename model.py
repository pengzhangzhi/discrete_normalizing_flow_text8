import torch
import torch.nn as nn
import torch.nn.functional as F

from tarflow_single_system import TarFlow


def apply_bert_mask(
    x_onehot: torch.Tensor, mask_ratio: float = 0.15
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply BERT-style masking by zeroing out random positions.
    
    Args:
        x_onehot: (B, T, vocab_size) one-hot encoded input
        mask_ratio: fraction of positions to mask
    
    Returns:
        x_masked: (B, T, vocab_size) with masked positions zeroed
        mask: (B, T) boolean tensor, True at masked positions
    """
    B, T, V = x_onehot.shape
    # Sample mask positions
    mask = torch.rand(B, T, device=x_onehot.device) < mask_ratio
    # Zero out masked positions
    x_masked = x_onehot.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


class MLMHead(nn.Module):
    """Simple MLM prediction head: LayerNorm -> Linear."""
    
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, hidden_dim) -> logits: (B, T, vocab_size)"""
        return self.proj(self.ln(x))


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
    
    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """x: (B, seq_len, vocab_size) one-hot -> (B, seq_len, hidden_dim)"""
        x = self.input_proj(x) + self.pos_embed
        hiddens = [] if return_intermediates else None
        for block in self.blocks:
            x = block(x)
            if return_intermediates:
                hiddens.append(x)
        out = self.ln_out(x)
        if return_intermediates:
            return out, hiddens
        return out


class TarFlowModel(nn.Module):
    """Encoder + TarFlow normalizing flow with shared hidden dimension and optional MLM."""
    
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
        noise_std_ratio: float = 0.0,
        mlm_enabled: bool = False,
    ):
        super().__init__()
        self.noise_std_ratio = noise_std_ratio
        self.mlm_enabled = mlm_enabled
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
        if mlm_enabled:
            self.mlm_head = MLMHead(hidden_dim, vocab_size)
    
    def forward(
        self, x: torch.Tensor, return_encoder_output: bool = False, return_intermediates: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        x: (B, seq_len, vocab_size) -> z: (B, seq_len, hidden_dim), logdet: (B,)
        If return_encoder_output=True, also returns u: (B, seq_len, hidden_dim)
        If return_intermediates=True, returns list of hidden states from encoder + flow blocks
        """
        if return_intermediates:
            u, encoder_hiddens = self.encoder(x, return_intermediates=True)
        else:
            u = self.encoder(x)
        u_for_flow = u
        if self.training and self.noise_std_ratio > 0:
            u_for_flow = u + self.noise_std_ratio * u.std() * torch.randn_like(u)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            if return_intermediates:
                z, logdet, flow_hiddens = self.flow(u_for_flow.float(), return_intermediates=True)
                all_hiddens = encoder_hiddens + flow_hiddens
                return z, logdet, all_hiddens
            else:
                z, logdet = self.flow(u_for_flow.float())
        if return_encoder_output:
            return z, logdet, u
        return z, logdet
    
    def mlm_logits(self, u: torch.Tensor) -> torch.Tensor:
        """Compute MLM logits from encoder output."""
        return self.mlm_head(u)
    
    def reverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reverse the flow: latent -> embeddings."""
        with torch.amp.autocast(device_type="cuda", enabled=False):
            return self.flow.reverse(z.float())


class Generator(nn.Module):
    """BERT-style generator: Z → logits.
    
    Takes Gaussian noise Z as input and outputs token logits.
    Architecture matches the encoder (encoder_layers + flow_blocks total blocks).
    """
    
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden_dim: int,
        n_blocks: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        self.blocks = nn.ModuleList([
            Block(hidden_dim, n_heads, dropout=dropout) for _ in range(n_blocks)
        ])
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
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
    
    def forward(self, z: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """z: (B, seq_len, hidden_dim) → logits: (B, seq_len, vocab_size)"""
        x = z + self.pos_embed
        hiddens = [] if return_intermediates else None
        for block in self.blocks:
            x = block(x)
            if return_intermediates:
                hiddens.append(x)
        logits = self.output_proj(self.ln_out(x))
        if return_intermediates:
            return logits, hiddens
        return logits
    
    def sample(self, batch_size: int, device: torch.device, temperature: float = 1.0) -> torch.Tensor:
        """Sample tokens from Gaussian noise."""
        z = torch.randn(batch_size, self.seq_len, self.hidden_dim, device=device)
        logits = self.forward(z)
        if temperature == 0:
            tokens = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, self.seq_len)
        return tokens
