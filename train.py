"""Unified training script for NFEncoder (Stage A) and Generator (Stage B).

Usage:
    # Train encoder (Stage A)
    python train.py +experiment=train_encoder
    
    # Train generator (Stage B)
    python train.py +experiment=train_generator training.encoder_ckpt=path/to/encoder.ckpt
    
    # Override model params
    python train.py +experiment=train_encoder model.hidden_dim=512 model.encoder_layers=8
"""

import json
from datetime import timedelta
from pathlib import Path

import hydra
import pytorch_lightning as pl
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def get_or_create_wandb_run_id(save_dir: Path) -> str:
    """Get existing WandB run_id or create a new one."""
    run_id_file = save_dir / "wandb_run_id.json"
    if run_id_file.exists():
        return json.load(open(run_id_file))["run_id"]
    run_id = wandb.util.generate_id()
    save_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"run_id": run_id}, open(run_id_file, "w"))
    return run_id


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print resolved config
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)
    
    # Setup directories
    save_dir = Path(cfg.logging.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Instantiate data module
    datamodule = instantiate(cfg.data)
    
    # Instantiate model
    model = instantiate(cfg.model)
    
    # Instantiate Lightning module (training logic)
    # Use _recursive_=False to prevent re-instantiation of nested _target_ in cfg
    lightning_module = instantiate(cfg.training, model=model, cfg=cfg, _recursive_=False)
    
    # Checkpoint and resume setup
    ckpt_path = save_dir / "last.ckpt"
    is_resume = ckpt_path.exists()
    wandb_run_id = get_or_create_wandb_run_id(save_dir)
    
    print(f">>> {'Resuming' if is_resume else 'Starting fresh'} | ckpt={ckpt_path if is_resume else 'None'} | wandb_id={wandb_run_id}")
    
    # Setup trainer with callbacks and logger (all settings from cfg.trainer)
    trainer = pl.Trainer(
        max_steps=cfg.trainer.max_steps,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        val_check_interval=cfg.trainer.val_check_interval,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        strategy=cfg.trainer.strategy,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        logger=WandbLogger(
            project=cfg.logging.project,
            name=cfg.logging.run_name,
            save_dir=str(save_dir),
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
    )
    
    # Train
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=ckpt_path if is_resume else None)
    
    # Always save final checkpoint
    trainer.save_checkpoint(save_dir / "last.ckpt")
    print(f">>> Saved final checkpoint to {save_dir / 'last.ckpt'}")


if __name__ == "__main__":
    main()
