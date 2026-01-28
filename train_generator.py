"""Stage B: Train BERT-style generator to map Gaussian noise to sequences.

Uses a frozen Stage A encoder to create (X, Z) pairs, then trains a generator
to reconstruct X from Z with representation alignment.
"""
import json
import math
import pickle
from datetime import timedelta
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import Text8DataModule
from model import Generator, TarFlowModel


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_encoder_from_checkpoint(ckpt_path: str, cfg: DictConfig) -> TarFlowModel:
    """Load a frozen Stage A encoder from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Extract model hyperparameters from checkpoint
    hparams = ckpt.get("hyper_parameters", {})
    
    encoder = TarFlowModel(
        vocab_size=hparams.get("model", {}).get("vocab_size", cfg.model.vocab_size),
        seq_len=hparams.get("data", {}).get("seq_len", cfg.data.seq_len),
        hidden_dim=hparams.get("model", {}).get("hidden_dim", cfg.model.hidden_dim),
        encoder_layers=hparams.get("encoder", {}).get("n_layers", cfg.encoder.n_layers),
        encoder_heads=hparams.get("encoder", {}).get("n_heads", cfg.encoder.n_heads),
        flow_blocks=hparams.get("flow", {}).get("n_blocks", cfg.flow.n_blocks),
        flow_layers_per_block=hparams.get("flow", {}).get("layers_per_block", cfg.flow.layers_per_block),
        dropout=0.0,  # No dropout for frozen encoder
        noise_std_ratio=0.0,  # No noise for frozen encoder
        mlm_enabled=True,
    )
    
    # Load state dict (Lightning wraps model in "model." prefix)
    state_dict = ckpt["state_dict"]
    encoder_state = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
    encoder.load_state_dict(encoder_state)
    
    # Freeze encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    return encoder


def load_itos(data_root: str) -> dict[int, str]:
    """Load index-to-string mapping for decoding."""
    meta_path = Path(data_root) / "meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta["itos"]


class GeneratorLightning(pl.LightningModule):
    """PyTorch Lightning module for Stage B generator training."""
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg
        
        # Load frozen encoder
        encoder_ckpt = cfg.generator.encoder_ckpt
        if encoder_ckpt is None:
            raise ValueError("generator.encoder_ckpt must be specified for Stage B training")
        self.encoder = load_encoder_from_checkpoint(encoder_ckpt, cfg)
        
        # Generator: total blocks = encoder blocks + flow blocks
        n_blocks = cfg.encoder.n_layers + cfg.flow.n_blocks
        self.generator = Generator(
            vocab_size=cfg.model.vocab_size,
            seq_len=cfg.data.seq_len,
            hidden_dim=cfg.model.hidden_dim,
            n_blocks=n_blocks,
            n_heads=cfg.encoder.n_heads,
            dropout=cfg.train.dropout,
        )
        
        # Lightweight adapters for representation alignment
        self.adapters = nn.ModuleList([
            nn.Linear(cfg.model.hidden_dim, cfg.model.hidden_dim) for _ in range(n_blocks)
        ])
        
        # Loss weights
        self.align_weight = cfg.generator.align_weight
        
        # Load itos for decoding samples
        self.itos = load_itos(cfg.data.root)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x_onehot, x_indices = batch
        
        # Encode with frozen teacher (no grad)
        with torch.no_grad():
            z, _, teacher_hiddens = self.encoder(x_onehot, return_intermediates=True)
        
        # Generate
        logits, student_hiddens = self.generator(z, return_intermediates=True)
        
        # Reconstruction loss (cross-entropy)
        loss_rec = F.cross_entropy(
            logits.view(-1, self.cfg.model.vocab_size),
            x_indices.view(-1),
        )
        
        # Representation alignment loss
        loss_align = torch.tensor(0.0, device=self.device)
        for adapter, h_s, h_t in zip(self.adapters, student_hiddens, teacher_hiddens):
            loss_align = loss_align + F.mse_loss(adapter(h_s), h_t.detach())
        loss_align = loss_align / len(self.adapters)
        
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
        
        # Encode with frozen teacher
        with torch.no_grad():
            z, _, _ = self.encoder(x_onehot, return_intermediates=True)
        
        # Generate
        logits = self.generator(z)
        
        # Reconstruction loss
        loss = F.cross_entropy(
            logits.view(-1, self.cfg.model.vocab_size),
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
        
        return loss
    
    def _log_samples(self, num_samples: int = 8):
        """Sample from Gaussian and log generated text to W&B."""
        z = torch.randn(num_samples, self.cfg.data.seq_len, self.cfg.model.hidden_dim, device=self.device)
        logits = self.generator(z)
        tokens = logits.argmax(dim=-1)  # (num_samples, seq_len)
        
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
            self.logger.experiment.log({"samples": table, "global_step": self.global_step})
    
    def configure_optimizers(self):
        # Only optimize generator and adapters (encoder is frozen)
        params = list(self.generator.parameters()) + list(self.adapters.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=self.cfg.train.warmup_steps,
            total_steps=self.cfg.train.max_steps,
            min_lr_ratio=self.cfg.train.min_lr_ratio,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def get_or_create_wandb_run_id(save_dir: Path) -> str:
    """Get existing WandB run_id or create a new one. Persists to save_dir for resume."""
    run_id_file = save_dir / "wandb_run_id.json"
    if run_id_file.exists():
        return json.load(open(run_id_file))["run_id"]
    run_id = wandb.util.generate_id()
    json.dump({"run_id": run_id}, open(run_id_file, "w"))
    return run_id


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.train.seed)
    
    save_dir = Path(cfg.logging.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    datamodule = Text8DataModule(
        root=cfg.data.root,
        batch_size=cfg.data.batch_size,
        seq_len=cfg.data.seq_len,
        num_workers=cfg.data.num_workers,
    )
    model = GeneratorLightning(cfg)
    
    # Checkpoint and resume setup
    ckpt_path = save_dir / "last.ckpt"
    is_resume = ckpt_path.exists()
    
    wandb_run_id = get_or_create_wandb_run_id(save_dir)
    
    print(f">>> {'Resuming' if is_resume else 'Starting fresh'} | ckpt={ckpt_path if is_resume else 'None'} | wandb_id={wandb_run_id}")
    print(f">>> Encoder checkpoint: {cfg.generator.encoder_ckpt}")
    
    trainer = pl.Trainer(
        max_steps=cfg.train.max_steps,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.grad_clip,
        val_check_interval=cfg.train.val_check_interval,
        log_every_n_steps=cfg.train.log_every_n_steps,
        logger=WandbLogger(
            project=cfg.logging.project,
            name=cfg.logging.run_name,
            save_dir=cfg.logging.save_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
            id=wandb_run_id,
            resume="allow",
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=save_dir,
                filename="last",
                save_last=False,
                train_time_interval=timedelta(minutes=10),
                save_top_k=1,
                enable_version_counter=False,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        enable_progress_bar=True,
        strategy=cfg.distributed.strategy,
        devices=cfg.distributed.devices,
        num_nodes=cfg.distributed.num_nodes,
    )
    
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path if is_resume else None)


if __name__ == "__main__":
    main()
