from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import get_dataloaders
from model import TarFlowModel


class TarFlowLightning(pl.LightningModule):
    """PyTorch Lightning module for TarFlow training."""
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg
        
        self.model = TarFlowModel(
            vocab_size=cfg.encoder.vocab_size,
            seq_len=cfg.data.seq_len,
            encoder_hidden=cfg.encoder.hidden_dim,
            encoder_output=cfg.encoder.output_dim,
            encoder_layers=cfg.encoder.n_layers,
            encoder_heads=cfg.encoder.n_heads,
            flow_hidden=cfg.flow.hidden_dim,
            flow_blocks=cfg.flow.n_blocks,
            flow_layers_per_block=cfg.flow.layers_per_block,
            dropout=cfg.train.dropout,
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x = batch[0]
        z, logdet = self.model(x)
        
        # MLE loss: -log p(x) = 0.5 * ||z||^2 - logdet
        log_pz = -0.5 * z.pow(2).sum(dim=[1, 2])
        nll = -(log_pz + logdet).mean()
        
        self.log("train/loss", nll, prog_bar=True, sync_dist=True)
        self.log("train/logdet", logdet.mean(), sync_dist=True)
        self.log("train/z_norm", z.pow(2).mean().sqrt(), sync_dist=True)
        return nll
    
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x = batch[0]
        z, logdet = self.model(x)
        
        log_pz = -0.5 * z.pow(2).sum(dim=[1, 2])
        nll = -(log_pz + logdet).mean()
        
        self.log("val/loss", nll, prog_bar=True, sync_dist=True)
        return nll
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
            betas=(0.9, 0.999),
        )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print config
    print(OmegaConf.to_yaml(cfg))
    
    # Seed
    pl.seed_everything(cfg.train.seed)
    
    # Data
    train_loader, val_loader = get_dataloaders(
        root=cfg.data.root,
        batch_size=cfg.data.batch_size,
        seq_len=cfg.data.seq_len,
        num_workers=cfg.data.num_workers,
    )
    
    # Model
    model = TarFlowLightning(cfg)
    
    # Logger
    wandb_logger = WandbLogger(
        project=cfg.logging.project,
        name=cfg.logging.run_name,
        save_dir=cfg.logging.save_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    
    # Callbacks
    save_dir = Path(cfg.logging.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="tarflow-{step:06d}-{val/loss:.4f}",
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    
    # Trainer
    trainer = pl.Trainer(
        max_steps=cfg.train.max_steps,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.grad_clip,
        val_check_interval=cfg.train.val_check_interval,
        log_every_n_steps=cfg.train.log_every_n_steps,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        enable_progress_bar=True,
        strategy=cfg.distributed.strategy,
        devices=cfg.distributed.devices,
        num_nodes=cfg.distributed.num_nodes,
    )
    
    # Train
    ckpt_path = cfg.get("resume", None)
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
