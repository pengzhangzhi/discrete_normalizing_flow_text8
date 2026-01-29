"""NFEncoder: Normalizing Flow Encoder for discrete sequences.

Maps input sequences X to Gaussian latent Z via:
1. TextEncoder: X (one-hot) -> U (embeddings)
2. TarFlow: U -> Z (Gaussian)

Always uses RoPE (Rotary Positional Encoding) for flexible sequence length handling.
"""

import torch
import torch.nn as nn

from .components import MLMHead, TarFlow, TextEncoder


class NFEncoder(nn.Module):
    """Normalizing Flow Encoder: X -> Z where Z ~ N(0, I).
    
    Combines a BERT-style text encoder (with RoPE) and a TarFlow normalizing flow.
    """
    
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,  # Required for TarFlow img_size calculation
        hidden_dim: int,
        encoder_layers: int,
        encoder_heads: int,
        flow_blocks: int,
        flow_layers_per_block: int = 1,
        dropout: float = 0.0,
        noise_std_ratio: float = 0.0,
        mlm_enabled: bool = False,
        ae_enabled: bool = False,
    ):
        super().__init__()
        # Store hyperparameters for checkpoint reconstruction
        self.hparams = {
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "encoder_layers": encoder_layers,
            "encoder_heads": encoder_heads,
            "flow_blocks": flow_blocks,
            "flow_layers_per_block": flow_layers_per_block,
            "dropout": dropout,
            "noise_std_ratio": noise_std_ratio,
            "mlm_enabled": mlm_enabled,
            "ae_enabled": ae_enabled,
        }
        
        self.noise_std_ratio = noise_std_ratio
        self.mlm_enabled = mlm_enabled
        self.ae_enabled = ae_enabled
        
        self.encoder = TextEncoder(
            vocab_size=vocab_size,
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
        
        if ae_enabled:
            self.ae_head = MLMHead(hidden_dim, vocab_size)
    
    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: str = "cpu") -> "NFEncoder":
        """Load NFEncoder from a Lightning checkpoint.
        
        Automatically extracts hyperparameters and weights from the checkpoint.
        """
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Extract hyperparameters from checkpoint (config stored under cfg key)
        hparams = ckpt.get("hyper_parameters", {})
        cfg = hparams.get("cfg", {})
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        
        state_dict = ckpt["state_dict"]
        model_state = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        
        # Build model with checkpoint hyperparameters
        model = cls(
            vocab_size=model_cfg.get("vocab_size", 27),
            seq_len=data_cfg.get("seq_len", 256),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            encoder_layers=model_cfg.get("encoder_layers", 4),
            encoder_heads=model_cfg.get("encoder_heads", 8),
            flow_blocks=model_cfg.get("flow_blocks", 4),
            flow_layers_per_block=model_cfg.get("flow_layers_per_block", 1),
            dropout=0.0,  # No dropout for inference
            noise_std_ratio=0.0,  # No noise for inference
            mlm_enabled=model_cfg.get("mlm_enabled", False),
        )
        
        # Load state dict
        model.load_state_dict(model_state)
        
        return model
    
    def forward(
        self, x: torch.Tensor, return_encoder_output: bool = False, return_intermediates: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Forward pass: X -> Z.
        
        Args:
            x: (B, seq_len, vocab_size) one-hot encoded input
            return_encoder_output: If True, also returns encoder embeddings U
            return_intermediates: If True, returns list of hidden states from encoder + flow
        
        Returns:
            z: (B, seq_len, hidden_dim) latent representation
            logdet: (B,) log determinant of Jacobian
            [optional] u or hiddens depending on flags
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
    
    def ae_logits(self, u: torch.Tensor) -> torch.Tensor:
        """Compute autoencoding logits from encoder output."""
        return self.ae_head(u)
    
    def reverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reverse the flow: Z -> U (embeddings)."""
        with torch.amp.autocast(device_type="cuda", enabled=False):
            return self.flow.reverse(z.float())
    
    def get_num_intermediate_states(self) -> int:
        """Return the number of intermediate hidden states (for alignment)."""
        return self.hparams["encoder_layers"] + self.hparams["flow_blocks"]
