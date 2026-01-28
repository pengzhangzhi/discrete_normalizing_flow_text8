"""Generator: BERT-style model that maps Gaussian noise Z to token logits.

Independent architecture from NFEncoder - can have different hidden_dim, n_blocks, etc.
Always uses RoPE (Rotary Positional Encoding) for flexible sequence length handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import Block


class Generator(nn.Module):
    """BERT-style generator with RoPE: Z -> logits.
    
    Takes Gaussian noise Z as input and outputs token logits.
    Architecture is independent from the encoder.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        n_blocks: int,
        n_heads: int,
        dropout: float = 0.0,
        seq_len: int | None = None,  # Only needed for sampling
    ):
        super().__init__()
        
        # Store hyperparameters
        self.hparams = {
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "n_blocks": n_blocks,
            "n_heads": n_heads,
            "dropout": dropout,
        }
        
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        self.blocks = nn.ModuleList([
            Block(hidden_dim, n_heads, dropout=dropout) 
            for _ in range(n_blocks)
        ])
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: str = "cpu") -> "Generator":
        """Load Generator from a Lightning checkpoint.
        
        Automatically extracts hyperparameters and weights from the checkpoint.
        """
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Extract hyperparameters from checkpoint
        hparams = ckpt.get("hyper_parameters", {})
        model_cfg = hparams.get("model", {})
        data_cfg = hparams.get("data", {})
        
        state_dict = ckpt["state_dict"]
        model_state = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        
        # Build model with checkpoint hyperparameters
        model = cls(
            vocab_size=model_cfg.get("vocab_size", data_cfg.get("vocab_size", 27)),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            n_blocks=model_cfg.get("n_blocks", 8),
            n_heads=model_cfg.get("n_heads", 8),
            dropout=0.0,  # No dropout for inference
            seq_len=model_cfg.get("seq_len", data_cfg.get("seq_len", 256)),
        )
        
        # Load state dict
        model.load_state_dict(model_state)
        
        return model
    
    def forward(
        self, z: torch.Tensor, return_intermediates: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass: Z -> logits.
        
        Args:
            z: (B, seq_len, hidden_dim) Gaussian noise (or encoder latent)
            return_intermediates: If True, returns list of hidden states
        
        Returns:
            logits: (B, seq_len, vocab_size) token logits
            [optional] hiddens: list of hidden states from each block
        """
        x = z
        hiddens = [] if return_intermediates else None
        for block in self.blocks:
            x = block(x)
            if return_intermediates:
                hiddens.append(x)
        logits = self.output_proj(self.ln_out(x))
        if return_intermediates:
            return logits, hiddens
        return logits
    
    def sample(
        self, batch_size: int, device: torch.device, temperature: float = 1.0, seq_len: int | None = None
    ) -> torch.Tensor:
        """Sample tokens from Gaussian noise.
        
        Args:
            batch_size: Number of samples to generate
            device: Device to generate on
            temperature: Sampling temperature (0 = argmax, >0 = softmax sampling)
            seq_len: Sequence length to generate (uses self.seq_len if not provided)
        
        Returns:
            tokens: (batch_size, seq_len) sampled token indices
        """
        seq_len = seq_len or self.seq_len
        if seq_len is None:
            raise ValueError("seq_len must be provided for sampling when not set during init")
        z = torch.randn(batch_size, seq_len, self.hidden_dim, device=device)
        logits = self.forward(z)
        if temperature == 0:
            tokens = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, seq_len)
        return tokens
    
    def get_num_blocks(self) -> int:
        """Return the number of transformer blocks."""
        return len(self.blocks)
