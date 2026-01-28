"""GeneratorTraining: Lightning module for training Generator (Stage B)."""

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig

from src.data.text8 import load_itos
from src.models.generator import Generator
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


class GeneratorTraining(pl.LightningModule):
    """PyTorch Lightning module for Generator training.
    
    Uses a frozen NFEncoder to create (X, Z) pairs, then trains a Generator
    to reconstruct X from Z with optional representation alignment.
    
    Losses:
    - Reconstruction: Cross-entropy between predicted logits and true tokens
    - Alignment: MSE between generator and encoder hidden states (partial alignment)
    """
    
    def __init__(
        self,
        model: Generator,
        cfg: DictConfig,
        encoder_ckpt: str,
        lr: float = 3e-5,
        max_steps: int = 500000,
        warmup_steps: int = 10000,
        min_lr_ratio: float = 0.1,
        weight_decay: float = 1e-3,
        align_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.cfg = cfg
        
        # Load frozen encoder
        self.encoder = NFEncoder.from_checkpoint(encoder_ckpt)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Optimizer/scheduler params
        self.lr = lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.weight_decay = weight_decay
        self.align_weight = align_weight
        
        # Setup adapters for alignment (if align_weight > 0)
        if align_weight > 0:
            encoder_hidden_dim = self.encoder.hparams["hidden_dim"]
            generator_hidden_dim = model.hidden_dim
            n_encoder_states = self.encoder.get_num_intermediate_states()
            n_generator_states = model.get_num_blocks()
            
            # Align min(N_gen, N_enc) blocks
            self.n_align = min(n_encoder_states, n_generator_states)
            
            # Adapters project from generator hidden_dim to encoder hidden_dim
            self.adapters = nn.ModuleList([
                nn.Linear(generator_hidden_dim, encoder_hidden_dim) 
                for _ in range(self.n_align)
            ])
        else:
            self.n_align = 0
            self.adapters = None
        
        # Load itos for decoding samples
        self.itos = load_itos(cfg.data.root)
        self.vocab_size = cfg.data.vocab_size
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x_onehot, x_indices = batch
        
        # Encode with frozen encoder (no grad)
        with torch.no_grad():
            if self.align_weight > 0:
                z, _, teacher_hiddens = self.encoder(x_onehot, return_intermediates=True)
            else:
                z, _ = self.encoder(x_onehot)
                teacher_hiddens = None
        
        # Generate
        if self.align_weight > 0:
            logits, student_hiddens = self.model(z, return_intermediates=True)
        else:
            logits = self.model(z)
            student_hiddens = None
        
        # Reconstruction loss (cross-entropy)
        loss_rec = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            x_indices.view(-1),
        )
        
        # Representation alignment loss (partial alignment)
        if self.align_weight > 0 and self.adapters is not None:
            loss_align = torch.tensor(0.0, device=self.device)
            for i in range(self.n_align):
                h_s = student_hiddens[i]
                h_t = teacher_hiddens[i]
                loss_align = loss_align + F.mse_loss(self.adapters[i](h_s), h_t.detach())
            loss_align = loss_align / self.n_align
        else:
            loss_align = torch.tensor(0.0, device=self.device)
        
        # Combined loss
        loss = loss_rec + self.align_weight * loss_align
        
        # Logging
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/loss_rec", loss_rec, sync_dist=True)
        self.log("train/loss_align", loss_align, sync_dist=True)
        
        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == x_indices).float().mean()
            self.log("train/acc", acc, sync_dist=True)
        
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        grad_norm = sum(p.grad.norm(2).item() ** 2 for p in self.parameters() if p.grad is not None) ** 0.5
        self.log("train/grad_norm", grad_norm, sync_dist=True)
    
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x_onehot, x_indices = batch
        
        # Encode with frozen encoder
        with torch.no_grad():
            z, _ = self.encoder(x_onehot)
        
        # Generate
        logits = self.model(z)
        
        # Reconstruction loss
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            x_indices.view(-1),
        )
        
        # Accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == x_indices).float().mean()
        
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, sync_dist=True)
        
        # Sample and log text (only rank 0, first batch)
        if batch_idx == 0 and self.global_rank == 0:
            self._log_samples()
            self._log_reconstructions(x_indices, preds)
        
        return loss
    
    def _log_samples(self, num_samples: int = 8):
        """Sample from Gaussian and log generated text to W&B."""
        tokens = self.model.sample(num_samples, self.device, temperature=1.0)
        
        # Decode to text
        texts = []
        for seq in tokens:
            text = "".join([self.itos[t.item()] for t in seq])
            texts.append(text)
        
        # Log to W&B as a table
        if self.logger is not None:
            table = wandb.Table(columns=["sample_id", "generated_text"])
            for i, text in enumerate(texts):
                table.add_data(i, text)
            self.logger.experiment.log({"val/samples": table, "global_step": self.global_step})
            
            # Also log as plain text for quick viewing
            sample_text = "\n---\n".join([f"[{i}] {t}" for i, t in enumerate(texts)])
            self.logger.experiment.log({
                "val/samples_text": wandb.Html(f"<pre>{sample_text}</pre>"),
                "global_step": self.global_step
            })
    
    def _log_reconstructions(self, x_indices: torch.Tensor, preds: torch.Tensor, num_samples: int = 4):
        """Log reconstruction comparisons: ground truth vs predicted."""
        if self.logger is None:
            return
        
        # Take first num_samples from batch
        gt_tokens = x_indices[:num_samples]
        pred_tokens = preds[:num_samples]
        
        # Decode to text
        rows = []
        for i in range(num_samples):
            gt_text = "".join([self.itos[t.item()] for t in gt_tokens[i]])
            pred_text = "".join([self.itos[t.item()] for t in pred_tokens[i]])
            # Character-level accuracy for this sample
            match = (gt_tokens[i] == pred_tokens[i]).float().mean().item()
            rows.append((i, gt_text, pred_text, f"{match:.2%}"))
        
        # Log to W&B as a table
        table = wandb.Table(columns=["sample_id", "ground_truth", "reconstruction", "char_acc"])
        for row in rows:
            table.add_data(*row)
        self.logger.experiment.log({"val/reconstructions": table, "global_step": self.global_step})
    
    def configure_optimizers(self):
        # Only optimize generator and adapters (encoder is frozen)
        params = list(self.model.parameters())
        if self.adapters is not None:
            params += list(self.adapters.parameters())
        
        optimizer = torch.optim.AdamW(
            params,
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
