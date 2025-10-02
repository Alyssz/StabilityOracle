# train_masked_residue.py
import sys
sys.path.append('/repo/nunziati/StabilityOracle/StabilityOracle/model')

import os, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from blocks import Backbone
import pytorch_lightning as pl

from argparse import ArgumentParser
from functools import partial
import yaml
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import torchmetrics

from dataset import GraphTransformerDataset

class BackboneLightningModule(pl.LightningModule):
    def __init__(self, use_sadic=False, dropout=0.2, learning_rate=1e-4, weight_decay=0.01, compile=False, **backbone_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = Backbone(**backbone_kwargs)
        self.criterion = nn.CrossEntropyLoss()

        if compile and torch.__version__ >= "2.0":
            # PyTorch 2.0+ compile (optional, may fail in some environments)
            try:
                self.model = torch.compile(self.model)
                print("Model compilation enabled")
            except Exception as e:
                print(f"Model compilation failed: {e}")
                pass
 
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_sadic = use_sadic
 
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=20, top_k=1
        )
        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=20, top_k=1
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=20, top_k=1
        )
 
        self.train_acc_3 = torchmetrics.Accuracy(
            task="multiclass", num_classes=20, top_k=3
        )
        self.valid_acc_3 = torchmetrics.Accuracy(
            task="multiclass", num_classes=20, top_k=3
        )
        self.test_acc_3 = torchmetrics.Accuracy(
            task="multiclass", num_classes=20, top_k=3
        )

        self._use_znorm = False

    def inference(self, batch, batch_idx):
        inputs, targets = batch

        # Input validation - check for NaN/inf values
        for key, tensor in inputs.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                self.print(f"Warning: NaN/inf detected in input '{key}' at batch {batch_idx}")
                # Replace NaN/inf with zeros
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
                inputs[key] = tensor

        accessibility_key = "sadic" if self.use_sadic else "sasa"
        pp = torch.stack([inputs["charges"], inputs[accessibility_key]], dim=-1)
        ca = inputs["ca"]
        mask = inputs["mask"]
        
        # CORREZIONE FORSE NECESSARIA
        ca = ca.reshape(-1, 1, 3)
        mask = mask.bool()

        features, logits = self.model([None], inputs["atom_types"], pp, inputs["coords"], ca, mask)

        # Check outputs for NaN/inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            self.print(f"Warning: NaN/inf detected in model output at batch {batch_idx}")
            self.print(f"Logits stats: min={logits.min()}, max={logits.max()}, mean={logits.mean()}")

        return features, logits, targets

    def training_step(self, batch, batch_idx):
        features, logits, targets = self.inference(batch, batch_idx)

        loss = self.criterion(logits, targets)
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            self.print(f"NaN/inf loss detected at batch {batch_idx}")
            self.print(f"Logits stats: min={logits.min()}, max={logits.max()}, mean={logits.mean()}")
            self.print(f"Targets: {targets}")
            # Return a small loss to prevent training from crashing
            return torch.tensor(0.01, requires_grad=True, device=loss.device)
        
        self.train_acc(logits, targets)
        self.train_acc_3(logits, targets)

        self.log('train/loss', loss)
        self.log('train/acc', self.train_acc)
        self.log('train/acc_3', self.train_acc_3)

        return loss

    def validation_step(self, batch, batch_idx):
        features, logits, targets = self.inference(batch, batch_idx)

        loss = self.criterion(logits, targets)
        self.valid_acc(logits, targets)
        self.valid_acc_3(logits, targets)

        self.log('val/loss', loss)
        self.log('val/acc', self.valid_acc)
        self.log('val/acc_3', self.valid_acc_3)

        return loss

    def test_step(self, batch, batch_idx):
        features, logits, targets = self.inference(batch, batch_idx)

        loss = self.criterion(logits, targets)
        self.test_acc(logits, targets)
        self.test_acc_3(logits, targets)

        self.log('test/loss', loss)
        self.log('test/acc', self.test_acc)
        self.log('test/acc_3', self.test_acc_3)

        return loss

    def on_fit_start(self):
        # --- W&B: record gradients/parameter histograms (lightweight, adjust log_freq if needed)
        if isinstance(self.logger, WandbLogger):
            self.logger.watch(self.model, log="all", log_freq=200)
            run = self.logger.experiment
            run.define_metric("val/acc", summary="max", step_metric="trainer/global_step")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LiteModel")
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-2)
        parser.add_argument("--max_epochs", type=int, default=40)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--compile", action="store_true", help="Enable torch.compile (may cause issues in some environments)")
        return parent_parser
 
    def configure_optimizers(self):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or "bias" in n or "norm" in n or "ln" in n or "layer_norm" in n:
                no_decay.append(p)
            else:
                decay.append(p)
 
        optim_kwargs = dict(lr=self.learning_rate, weight_decay=self.weight_decay, eps=1e-8)
        # fused AdamW if available (PyTorch 2+ with recent CUDA)
        try:
            optimizer = torch.optim.AdamW(
                [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}],
                fused=True, **optim_kwargs
            )
        except TypeError:
            optimizer = torch.optim.AdamW(
                [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}],
                **optim_kwargs
            )
 
        # Warmup + cosine (step-wise)
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None) or 1000
        warmup_steps = max(int(0.06 * total_steps), 1)
 
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
 
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "interval": "step",          # update every step
            #     "frequency": 1,
            # },
        }

def run_training(args): 
    # --- W&B: configure mode
    if args.wandb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
    elif args.wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
 
    # --- W&B: init Lightning logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        save_dir="./wandb",
        log_model=True,                 # upload best/last checkpoints as artifacts
        id=args.wandb_resume_id,        # to resume specific run
        resume="allow" if args.wandb_resume_id else None,
    )
 
    lr_monitor = LearningRateMonitor(logging_interval="step")
 
    dataset = GraphTransformerDataset(args.data_path, max_entries=args.max_entries)
    # Create random splits for train, validation, and test
    num_examples = len(dataset)
    indices = list(range(num_examples))
    random.shuffle(indices)

    train_split = int(0.8 * num_examples)
    valid_split = int(0.9 * num_examples)

    train_indices = indices[:train_split]
    valid_indices = indices[train_split:valid_split]
    test_indices = indices[valid_split:]

    train_data = Subset(dataset, train_indices)
    valid_data = Subset(dataset, valid_indices)
    test_data = Subset(dataset, test_indices)
 
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        dirpath="./checkpoints/",
        filename="model-{epoch:02d}-{val_loss:.3f}",
    )
 
    early_stopping_callback = EarlyStopping(
        monitor="val/acc",
        patience=20,
        mode="max",
    )
 
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="32",  # Use full precision to avoid numerical instability
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping_callback],
        logger=wandb_logger,  # --- W&B: hook logger into Trainer
        gradient_clip_val=1.0,  # Add gradient clipping to prevent NaN
        gradient_clip_algorithm="norm",
    )
 
    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True, collate_fn=dataset.collate_fn
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True, collate_fn=dataset.collate_fn
    )

 
    litmodel = BackboneLightningModule(
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_sadic=args.enable_sadic,
        compile=args.compile,
    )
 
    # # z-norm: compute or load stats, then set them on the model
    # if args.z_norm:
    #     os.makedirs(os.path.dirname(args.zstats_path), exist_ok=True)
    #     if os.path.exists(args.zstats_path):
    #         stats = torch.load(args.zstats_path, map_location="cpu")
    #         node_mean = stats["node_mean"]; node_std = stats["node_std"]
    #         edge_mean = stats["edge_mean"]; edge_std = stats["edge_std"]
    #         print(f"Loaded z-norm stats from {args.zstats_path}")
    #     else:
    #         # Important: use the training loader only (no shuffling required here)
    #         # You can create a non-shuffling loader to avoid repeated data if you want
    #         node_mean, node_std, edge_mean, edge_std = compute_feature_stats_from_loader(train_loader)
    #         torch.save(
    #             {"node_mean": node_mean, "node_std": node_std, "edge_mean": edge_mean, "edge_std": edge_std},
    #             args.zstats_path,
    #         )
    #         print(f"Saved z-norm stats to {args.zstats_path}")
    #     # attach to model (as buffers) so they move with .to(device) and are stored in checkpoints
    #     litmodel.set_normalizer(node_mean, node_std, edge_mean, edge_std)
    #     # Optional: log to W&B for visibility
    #     if isinstance(wandb_logger, WandbLogger):
    #         wandb_logger.experiment.summary["z_norm_enabled"] = True
    #         wandb_logger.experiment.summary["zstats_path"] = args.zstats_path
    # else:
    #     print("Z-normalization disabled. Run with --z_norm to enable.")
 
    trainer.fit(
        model=litmodel, train_dataloaders=train_loader, val_dataloaders=valid_loader)
 
    trainer.test(
        model=litmodel,
        ckpt_path="best",
        dataloaders=test_loader,
    )
 
    wandb.finish()

def main():
    torch.set_float32_matmul_precision("high")
 
    parser = ArgumentParser()
    parser = BackboneLightningModule.add_model_specific_args(parser)
    parser.add_argument("--data_path", type=str, default="/repo/nunziati/StabilityOracle/frank/data/masked_prediction/training_structural_info_with_SADIC.hdf5")
 
    # after your existing parser args:
    # parser.add_argument("--z_norm", action="store_true", help="Enable z-normalization of node/edge features based on the training set")
    # parser.add_argument("--zstats_path", type=str, default="./checkpoints/znorm.pt", help="Path to save/load z-norm stats (torch.save)")
 
    parser.add_argument("--enable_sadic", action="store_true")
    parser.add_argument("--max_entries", type=int, default=0, help="Maximum number of entries to load from the dataset (for debugging). 0 means all.")
 
    # --- W&B specific flags
    parser.add_argument("--wandb_project", type=str, default="stability-oracle")
    parser.add_argument("--wandb_entity", type=str, default=None)  # your team/org, or None
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_resume_id", type=str, default=None)  # resume by run id if set
    parser.add_argument("--sweep_config", type=str, default=None, help="Path to W&B sweep config YAML/JSON")
    parser.add_argument("--sweep_id", type=str, default=None, help="W&B sweep id to resume")
 
    args = parser.parse_args()

    def sweep_main(run_args):
        wandb.init()
        vars(run_args).update(dict(wandb.config))
        run_training(run_args)
 
    if args.sweep_config is not None:
        # Start a new sweep
        with open(args.sweep_config, "r") as f:
            sweep_config = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
        wandb.agent(sweep_id, function=partial(sweep_main, args))
    elif args.sweep_id is not None:
        # Resume an existing sweep
        wandb.agent(args.sweep_id, function=partial(sweep_main, args))
    else:
        # Single run
        run_training(args)

if __name__ == "__main__":
    main()
