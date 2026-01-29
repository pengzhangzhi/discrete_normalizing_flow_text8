"""EncoderTraining: Lightning module for training NFEncoder (Stage A)."""

import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.models.components import apply_bert_mask
from src.models.nf_encoder import NFEncoder


def get_cosine_schedule_with_warmup(
    optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1
):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EncoderTraining(pl.LightningModule):
    """PyTorch Lightning module for NFEncoder training.
    
    Trains the normalizing flow encoder with:
    - Flow NLL loss: -log p(x) = 0.5 * ||z||^2 - logdet
    - Optional MLM auxiliary loss
    """
    
    def __init__(
        self,
        model: NFEncoder,
        cfg: DictConfig,
        lr: float = 3e-5,
        max_steps: int = 500000,
        warmup_steps: int = 10000,
        min_lr_ratio: float = 0.1,
        weight_decay: float = 1e-3,
        mlm_enabled: bool = False,
        mlm_mask_ratio: float = 0.15,
        mlm_weight: float = 1.0,
        ae_loss_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.cfg = cfg
        
        # Optimizer/scheduler params
        self.lr = lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.weight_decay = weight_decay
        
        # MLM params
        self.mlm_enabled = mlm_enabled
        self.mlm_mask_ratio = mlm_mask_ratio
        self.mlm_weight = mlm_weight
        
        # Autoencoding loss params
        self.ae_loss_weight = ae_loss_weight
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x_onehot, x_indices = batch
        
        # Determine if we need encoder output u
        need_encoder_output = self.mlm_enabled or self.ae_loss_weight > 0
        
        if self.mlm_enabled:
            # Apply BERT-style masking
            x_masked, mask = apply_bert_mask(x_onehot, self.mlm_mask_ratio)
            z, logdet, u = self.model(x_masked, return_encoder_output=True)
            
            # MLM loss: cross-entropy on masked positions only
            mlm_logits = self.model.mlm_logits(u)
            mlm_loss = F.cross_entropy(
                mlm_logits[mask].view(-1, mlm_logits.size(-1)),
                x_indices[mask].view(-1),
            )
        elif need_encoder_output:
            z, logdet, u = self.model(x_onehot, return_encoder_output=True)
            mlm_loss = torch.tensor(0.0, device=z.device)
        else:
            z, logdet = self.model(x_onehot)
            mlm_loss = torch.tensor(0.0, device=z.device)
        
        # Autoencoding loss: cross-entropy on all positions to reconstruct x from u
        if self.ae_loss_weight > 0:
            ae_logits = self.model.ae_logits(u)
            ae_loss = F.cross_entropy(
                ae_logits.view(-1, ae_logits.size(-1)),
                x_indices.view(-1),
            )
        else:
            ae_loss = torch.tensor(0.0, device=z.device)
        
        # Flow NLL: -log p(x) = 0.5 * ||z||^2 - logdet
        log_pz = -0.5 * z.pow(2).mean(dim=[1, 2])
        flow_loss = -(log_pz + logdet).mean()
        
        # Combined loss
        loss = flow_loss + self.mlm_weight * mlm_loss + self.ae_loss_weight * ae_loss
        
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/flow_loss", flow_loss, sync_dist=True)
        self.log("train/mlm_loss", mlm_loss, sync_dist=True)
        self.log("train/ae_loss", ae_loss, sync_dist=True)
        self.log("train/logdet", logdet.mean(), sync_dist=True)
        self.log("train/z_norm", z.pow(2).mean().sqrt(), sync_dist=True)
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        grad_norm = sum(p.grad.norm(2).item() ** 2 for p in self.parameters() if p.grad is not None) ** 0.5
        self.log("train/grad_norm", grad_norm, sync_dist=True)
    
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x_onehot, x_indices = batch
        z, logdet = self.model(x_onehot)
        
        log_pz = -0.5 * z.pow(2).mean(dim=[1, 2])
        flow_loss = -(log_pz + logdet).mean()
        
        self.log("val/loss", flow_loss, prog_bar=True, sync_dist=True)
        return flow_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.max_steps,
            min_lr_ratio=self.min_lr_ratio,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
