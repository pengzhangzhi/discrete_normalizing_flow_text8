import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import get_dataloaders
from model import TarFlowModel


class TarFlowLightning(pl.LightningModule):
    """PyTorch Lightning module for TarFlow training."""
    
    def __init__(
        self,
        vocab_size: int = 27,
        seq_len: int = 256,
        encoder_hidden: int = 256,
        encoder_output: int = 64,
        encoder_layers: int = 2,
        encoder_heads: int = 8,
        flow_hidden: int = 768,
        flow_blocks: int = 12,
        flow_layers_per_block: int = 1,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = TarFlowModel(
            vocab_size=vocab_size,
            seq_len=seq_len,
            encoder_hidden=encoder_hidden,
            encoder_output=encoder_output,
            encoder_layers=encoder_layers,
            encoder_heads=encoder_heads,
            flow_hidden=flow_hidden,
            flow_blocks=flow_blocks,
            flow_layers_per_block=flow_layers_per_block,
            dropout=dropout,
        )
        self.lr = lr
        self.weight_decay = weight_decay
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x = batch[0]  # (B, seq_len, vocab_size)
        z, logdet = self.model(x)
        
        # MLE loss: -log p(x) = 0.5 * ||z||^2 - logdet
        log_pz = -0.5 * z.pow(2).sum(dim=[1, 2])  # (B,)
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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        return optimizer


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train TarFlow on Text8")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    
    # Allow CLI overrides for common params
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--devices", type=str, default=None, help="Number of GPUs or 'auto'")
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--strategy", type=str, default=None, help="auto, ddp, fsdp, deepspeed")
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # CLI overrides
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.devices is not None:
        cfg["distributed"]["devices"] = args.devices if args.devices == "auto" else int(args.devices)
    if args.num_nodes is not None:
        cfg["distributed"]["num_nodes"] = args.num_nodes
    if args.strategy is not None:
        cfg["distributed"]["strategy"] = args.strategy
    if args.precision is not None:
        cfg["train"]["precision"] = args.precision
    if args.run_name is not None:
        cfg["logging"]["run_name"] = args.run_name
    
    # Parse devices (handle "auto" vs int vs list)
    devices = cfg["distributed"]["devices"]
    if isinstance(devices, str) and devices != "auto":
        devices = int(devices)
    
    # Seed
    pl.seed_everything(cfg["train"]["seed"])
    
    # Data
    train_loader, val_loader = get_dataloaders(
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        seq_len=cfg["data"]["seq_len"],
        num_workers=cfg["data"]["num_workers"],
    )
    
    # Model
    model = TarFlowLightning(
        vocab_size=cfg["encoder"]["vocab_size"],
        seq_len=cfg["data"]["seq_len"],
        encoder_hidden=cfg["encoder"]["hidden_dim"],
        encoder_output=cfg["encoder"]["output_dim"],
        encoder_layers=cfg["encoder"]["n_layers"],
        encoder_heads=cfg["encoder"]["n_heads"],
        flow_hidden=cfg["flow"]["hidden_dim"],
        flow_blocks=cfg["flow"]["n_blocks"],
        flow_layers_per_block=cfg["flow"]["layers_per_block"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        dropout=cfg["train"]["dropout"],
    )
    
    # Logger
    wandb_logger = WandbLogger(
        project=cfg["logging"]["project"],
        name=cfg["logging"]["run_name"],
        save_dir=cfg["logging"]["save_dir"],
        config=cfg,  # Log full config to wandb
    )
    
    # Callbacks
    save_dir = Path(cfg["logging"]["save_dir"])
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
    
    # Trainer with DDP support
    trainer = pl.Trainer(
        max_steps=cfg["train"]["max_steps"],
        precision=cfg["train"]["precision"],
        gradient_clip_val=cfg["train"]["grad_clip"],
        val_check_interval=cfg["train"]["val_check_interval"],
        log_every_n_steps=cfg["train"]["log_every_n_steps"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        enable_progress_bar=True,
        # DDP settings
        strategy=cfg["distributed"]["strategy"],
        devices=devices,
        num_nodes=cfg["distributed"]["num_nodes"],
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
