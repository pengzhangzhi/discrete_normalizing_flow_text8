import json
import math
from datetime import timedelta
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import Text8DataModule
from model import TarFlowModel


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class TarFlowLightning(pl.LightningModule):
    """PyTorch Lightning module for TarFlow training."""
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg
        
        self.model = TarFlowModel(
            vocab_size=cfg.model.vocab_size,
            seq_len=cfg.data.seq_len,
            hidden_dim=cfg.model.hidden_dim,
            encoder_layers=cfg.encoder.n_layers,
            encoder_heads=cfg.encoder.n_heads,
            flow_blocks=cfg.flow.n_blocks,
            flow_layers_per_block=cfg.flow.layers_per_block,
            dropout=cfg.train.dropout,
            noise_std=cfg.train.noise_std,
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x = batch[0]
        z, logdet = self.model(x)
        
        # NLL: -log p(x) = 0.5 * ||z||^2 - logdet
        log_pz = -0.5 * z.pow(2).mean(dim=[1, 2])
        nll = -(log_pz + logdet).mean()
        
        self.log("train/loss", nll, prog_bar=True, sync_dist=True)
        self.log("train/logdet", logdet.mean(), sync_dist=True)
        self.log("train/z_norm", z.pow(2).mean().sqrt(), sync_dist=True)
        return nll
    
    def on_before_optimizer_step(self, optimizer):
        grad_norm = sum(p.grad.norm(2).item() ** 2 for p in self.parameters() if p.grad is not None) ** 0.5
        self.log("train/grad_norm", grad_norm, sync_dist=True)
    
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x = batch[0]
        z, logdet = self.model(x)
        
        log_pz = -0.5 * z.pow(2).mean(dim=[1, 2])
        nll = -(log_pz + logdet).mean()
        
        self.log("val/loss", nll, prog_bar=True, sync_dist=True)
        return nll
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
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
    model = TarFlowLightning(cfg)
    
    # Checkpoint and resume setup
    ckpt_path = save_dir / "last.ckpt"
    is_resume = ckpt_path.exists()
    wandb_run_id = get_or_create_wandb_run_id(save_dir)
    
    print(f">>> {'Resuming' if is_resume else 'Starting fresh'} | ckpt={ckpt_path if is_resume else 'None'} | wandb_id={wandb_run_id}")
    
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
