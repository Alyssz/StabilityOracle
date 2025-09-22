import glob
import os
import pickle
from argparse import ArgumentParser
import yaml
from functools import partial
 
import pytorch_lightning as pl
import torch
import torchmetrics
from joblib import Parallel, delayed
from model import AMPNN
from pdb_utils import myDataset, parallel_converter
 
import numpy as np
import wandb  # --- W&B
from pytorch_lightning.loggers import WandbLogger  # --- W&B
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping  # add LR monitor
 
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
 
class Liteampnn(pl.LightningModule):
    def __init__(
        self,
        embed_dim=128,
        edge_dim=27,
        node_dim=28,
        dropout=0.2,
        layer_nums=3,
        token_num=21,
        learning_rate=1e-4,
        weight_decay=0.01,
        use_gat_logits=False,
    ) -> None:
        self.save_hyperparameters()
 
        super().__init__()
        self.ampnn = AMPNN(
            embed_dim=embed_dim,
            edge_dim=edge_dim,
            node_dim=node_dim,
            token_num=token_num,
            layer_nums=layer_nums,
            dropout=dropout,
            use_gat_logits=use_gat_logits,
        )
 
        try:
            self.ampnn = torch.compile(self.ampnn)
        except Exception:
            pass
 
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
 
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
 
        self.train_mrr = torchmetrics.retrieval.RetrievalMRR()
        self.valid_mrr = torchmetrics.retrieval.RetrievalMRR()
        self.test_mrr = torchmetrics.retrieval.RetrievalMRR()
 
        self._use_znorm = False
 
    # --- z-norm API ---
    def set_normalizer(self, node_mean, node_std, edge_mean, edge_std):
        # Clamp std to avoid div-by-zero, and register as buffers so they follow device/checkpoints
        node_mean = node_mean.float()
        node_std  = torch.clamp(node_std.float(), min=1e-6)
        edge_mean = edge_mean.float()
        edge_std  = torch.clamp(edge_std.float(), min=1e-6)
        self.register_buffer("node_mean", node_mean)
        self.register_buffer("node_std", node_std)
        self.register_buffer("edge_mean", edge_mean)
        self.register_buffer("edge_std", edge_std)
        self._use_znorm = True
 
    def _apply_znorm(self, node, edge):
        if self._use_znorm:
            node = (node - self.node_mean.to(node.device)) / self.node_std.to(node.device)
            if edge.numel() > 0:  # handle empty-edge cases
                edge = (edge - self.edge_mean.to(edge.device)) / self.edge_std.to(edge.device)
        return node, edge
 
    def on_fit_start(self):
        # --- W&B: record gradients/parameter histograms (lightweight, adjust log_freq if needed)
        if isinstance(self.logger, WandbLogger):
            self.logger.watch(self.ampnn, log="all", log_freq=200)
            run = self.logger.experiment
            run.define_metric("val_acc_step", summary="max", step_metric="trainer/global_step")
 
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Liteampnn")
        parser.add_argument("--embed_dim", type=int, default=128)
        parser.add_argument("--edge_dim", type=int, default=27)
        parser.add_argument("--node_dim", type=int, default=-1)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--layer_nums", type=int, default=3)
        parser.add_argument("--use_gat_logits", action="store_true")
        parser.add_argument("--token_num", type=int, default=21)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--max_epochs", type=int, default=40)
        parser.add_argument("--noise_level", type=float, default=0.5)
        return parent_parser
 
    def training_step(self, batch, batch_idx):
        node, edge, y = batch
        node, edge = self._apply_znorm(node, edge)
        y_hat, h = self.ampnn(node, edge)
        loss = nn.functional.cross_entropy(y_hat, y.squeeze(0))
        self.train_acc(y_hat, y.squeeze(0))
        self.train_acc_3(y_hat, y.squeeze(0))
 
        # B, C = y_hat.shape[:2]
        # y_hat_flat = y_hat.flatten()
        # target = torch.zeros_like(y_hat_flat)
        # target[torch.arange(B, device=y.device) * C + y.squeeze(0)] = 1
        # indexes = torch.arange(B, device=y.device).repeat_interleave(C)
        # self.train_mrr(y_hat_flat, target, indexes)
        # self.log("train_mrr_step", self.train_mrr)
 
        self.log("train_loss", loss)
        self.log("train_acc_step", self.train_acc)
        self.log("train_acc_3_step", self.train_acc_3)
        self.log("train_perplexity_step", torch.exp(loss))
        return loss
 
    def validation_step(self, batch, batch_idx):
        node, edge, y = batch
        node, edge = self._apply_znorm(node, edge)
        y_hat, h = self.ampnn(node, edge)
        val_loss = nn.functional.cross_entropy(y_hat, y.squeeze(0))
        self.valid_acc(y_hat, y.squeeze(0))
        self.valid_acc_3(y_hat, y.squeeze(0))
 
        # B, C = y_hat.shape[:2]
        # y_hat_flat = y_hat.flatten()
        # target = torch.zeros_like(y_hat_flat)
        # target[torch.arange(B, device=y.device) * C + y.squeeze(0)] = 1
        # indexes = torch.arange(B, device=y.device).repeat_interleave(C)
        # self.valid_mrr(y_hat_flat, target, indexes)
        # self.log("val_mrr_step", self.valid_mrr, sync_dist=True)
 
        self.log("val_loss", val_loss, sync_dist=True)
        self.log("val_acc_step", self.valid_acc, sync_dist=True)
        self.log("val_acc_3_step", self.valid_acc_3, sync_dist=True)
        self.log("val_perplexity_step", torch.exp(val_loss), sync_dist=True)
 
    def test_step(self, batch, batch_idx):
        node, edge, y = batch
        node, edge = self._apply_znorm(node, edge)
        y_hat, h = self.ampnn(node, edge)
        test_loss = nn.functional.cross_entropy(y_hat, y.squeeze(0))
        self.test_acc(y_hat, y.squeeze(0))
        self.test_acc_3(y_hat, y.squeeze(0))
 
        # B, C = y_hat.shape[:2]
        # y_hat_flat = y_hat.flatten()
        # target = torch.zeros_like(y_hat_flat)
        # target[torch.arange(B, device=y.device) * C + y.squeeze(0)] = 1
        # indexes = torch.arange(B, device=y.device).repeat_interleave(C)
        # self.test_mrr(y_hat_flat, target, indexes)
        # self.log("test_mrr_step", self.test_mrr, sync_dist=True)
 
        self.log("test_loss", test_loss)
        self.log("test_acc_step", self.test_acc)
        self.log("test_acc_3_step", self.test_acc_3)
        self.log("test_perplexity_step", torch.exp(test_loss))
 
    def configure_optimizers(self):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or "bias" in n or "norm" in n or "ln" in n or "layer_norm" in n:
                no_decay.append(p)
            else:
                decay.append(p)
 
        optim_kwargs = dict(lr=self.learning_rate, weight_decay=self.weight_decay)
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
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",          # update every step
                "frequency": 1,
            },
        }
 
 
def get_dataset(
    path: str,
    list_file=None,
    file_type="pkl", # in ["pdb", "pkl"]
    noise=0.0,
    neighbor=48,
    plus=False,
    enable_sadic=False,
):
    if list_file == None:
        if file_type == "pkl":
            all_pkls = glob.glob(os.path.join(path, "*.pkl"))
            all_Protbb = []
            for pkl in tqdm(all_pkls):
                protbb = pickle.load(open(pkl, "rb"))
                all_Protbb.append(protbb)
        if file_type == "pdb":
            all_pdbs = glob.glob(os.path.join(path, "*.pdb"))
            all_Protbb = Parallel(n_jobs=-1)(
                delayed(parallel_converter)(pdb) for pdb in tqdm(all_pdbs)
            )
    else:
        all_files = open(list_file, "r").read().split("\n")[:-1]
        if file_type == "pkl":
            all_Protbb = []
            for pkl in tqdm(all_files):
                protbb = pickle.load(open(pkl, "rb"))
                all_Protbb.append(protbb)
        if file_type == "pdb":
 
            all_Protbb = Parallel(n_jobs=-1)(
                delayed(parallel_converter)(pdb) for pdb in tqdm(all_files)
            )
            # all_Protbb = []
            # for pdb in tqdm(all_files):
            #     protbb = parallel_converter(pdb)
            #     all_Protbb.append(protbb)
 
    if plus:
        # dataset = myDatasetPlus(
        #     all_Protbb, noise=noise, neighbor=neighbor, meta_batchsize=1400
        # )
        pass
    else:
        dataset = myDataset(
            all_Protbb, noise=noise, neighbor=neighbor, meta_batchsize=2000, enable_sadic=enable_sadic
        )
 
    return dataset
 
@torch.no_grad()
def compute_feature_stats_from_loader(loader, limit=None):
    """
    Computes per-channel mean and std for node and edge features by streaming through the loader.
    Assumes each item is (node [N, Dn], edge [E, De], y).
    For the first 22 node features, sets mean=0 and std=1 (ignores them).
    """
    node_sum = node_sumsq = None
    edge_sum = edge_sumsq = None
    node_count = 0
    edge_count = 0
 
    # Fallback dims in case some graphs have empty edges
    edge_feat_dim = None
    node_feat_dim = None
 
    for i, (node, edge, _) in enumerate(tqdm(loader, desc="Computing z-norm stats")):
        if limit is not None and i >= limit:
            break
 
        node = node.double()
        n = node.shape[0] * node.shape[1]
        if node_feat_dim is None:
            node_feat_dim = node.shape[-1]
 
        if node_sum is None:
            node_sum = node.sum(dim=(0,1))
            node_sumsq = (node * node).sum(dim=(0,1))
        else:
            node_sum += node.sum(dim=(0,1))
            node_sumsq += (node * node).sum(dim=(0,1))
        node_count += n
 
        # Edge features (may be empty)
        if edge is not None and edge.numel() > 0:
            edge = edge.double()
            e = edge.shape[0] * edge.shape[1]
            if edge_feat_dim is None:
                edge_feat_dim = edge.shape[-1]
            if edge_sum is None:
                edge_sum = edge.sum(dim=(0,1))
                edge_sumsq = (edge * edge).sum(dim=(0,1))
            else:
                edge_sum += edge.sum(dim=(0,1))
                edge_sumsq += (edge * edge).sum(dim=(0,1))
            edge_count += e
 
    # Node stats
    node_mean = node_sum / max(node_count, 1)
    node_var = node_sumsq / max(node_count, 1) - node_mean * node_mean
    node_var = torch.clamp(node_var, min=1e-12)
    node_std = torch.sqrt(node_var)
 
    # Ignore first 22 node features: set mean=0, std=1
    if node_feat_dim is None:
        node_feat_dim = 28  # fallback
    node_mean[:22] = 0.0
    node_std[:22] = 1.0
 
    # Edge stats: if no edges observed, default to zeros/ones
    if edge_count > 0:
        edge_mean = edge_sum / edge_count
        edge_var = edge_sumsq / edge_count - edge_mean * edge_mean
        edge_var = torch.clamp(edge_var, min=1e-12)
        edge_std = torch.sqrt(edge_var)
    else:
        if edge_feat_dim is None:
            # best effort: infer from your model hyperparam (27)
            edge_feat_dim = 27
        edge_mean = torch.zeros(edge_feat_dim, dtype=torch.double)
        edge_std = torch.ones(edge_feat_dim, dtype=torch.double)
 
    return node_mean.float(), node_std.float(), edge_mean.float(), edge_std.float()
 
 
def run_training(args):
    if args.node_dim == -1:
        args.node_dim = 28 + (5 if args.enable_sadic else 0)
 
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
 
    train_data = get_dataset(
        args.train_data_dir, file_type="pdb", list_file=args.train_list_file, noise=args.noise_level, neighbor=32, enable_sadic=args.enable_sadic
    )
    test_data = get_dataset(
        args.test_data_dir, file_type="pdb", list_file=args.test_list_file, noise=0.00, neighbor=32, enable_sadic=args.enable_sadic
    )
 
    valid_data = get_dataset(
        args.valid_data_dir, file_type="pdb", list_file=args.valid_list_file, noise=0.00, neighbor=32, enable_sadic=args.enable_sadic
    )
 
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        dirpath="./checkpoints/",
        filename="model-{epoch:02d}-{val_loss:.3f}",
    )
 
    early_stopping_callback = EarlyStopping(
        monitor="val_acc_step",
        patience=20,
        mode="max",
    )
 
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision="16-mixed",
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping_callback],
        logger=wandb_logger,  # --- W&B: hook logger into Trainer
    )
 
    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_data, batch_size=None, shuffle=True, num_workers=0, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_data, batch_size=None, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=None, shuffle=False, num_workers=0, pin_memory=True
    )
 
 
    litmodel = Liteampnn(
        embed_dim=args.embed_dim,
        edge_dim=args.edge_dim,
        node_dim=args.node_dim,
        dropout=args.dropout,
        layer_nums=args.layer_nums,
        token_num=args.token_num,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_gat_logits=args.use_gat_logits,
    )
 
    # z-norm: compute or load stats, then set them on the model
    if args.z_norm:
        os.makedirs(os.path.dirname(args.zstats_path), exist_ok=True)
        if os.path.exists(args.zstats_path):
            stats = torch.load(args.zstats_path, map_location="cpu")
            node_mean = stats["node_mean"]; node_std = stats["node_std"]
            edge_mean = stats["edge_mean"]; edge_std = stats["edge_std"]
            print(f"Loaded z-norm stats from {args.zstats_path}")
        else:
            # Important: use the training loader only (no shuffling required here)
            # You can create a non-shuffling loader to avoid repeated data if you want
            node_mean, node_std, edge_mean, edge_std = compute_feature_stats_from_loader(train_loader)
            torch.save(
                {"node_mean": node_mean, "node_std": node_std, "edge_mean": edge_mean, "edge_std": edge_std},
                args.zstats_path,
            )
            print(f"Saved z-norm stats to {args.zstats_path}")
        # attach to model (as buffers) so they move with .to(device) and are stored in checkpoints
        litmodel.set_normalizer(node_mean, node_std, edge_mean, edge_std)
        # Optional: log to W&B for visibility
        if isinstance(wandb_logger, WandbLogger):
            wandb_logger.experiment.summary["z_norm_enabled"] = True
            wandb_logger.experiment.summary["zstats_path"] = args.zstats_path
    else:
        print("Z-normalization disabled. Run with --z_norm to enable.")
 
    trainer.fit(
        model=litmodel, train_dataloaders=train_loader, val_dataloaders=valid_loader)
 
    trainer.test(
        model=litmodel,
        ckpt_path="best",
        dataloaders=test_loader,
    )
 
    wandb.finish()
 
 
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
 
    parser = ArgumentParser()
    parser = Liteampnn.add_model_specific_args(parser)
    parser.add_argument("--train_data_dir", type=str, default="./train_data/")
    parser.add_argument("--test_data_dir", type=str, default="./test_data/")
    parser.add_argument("--valid_data_dir", type=str, default="./valid_data/")
    parser.add_argument("--train_list_file", type=str, default="./train_of_list.txt")
    parser.add_argument("--test_list_file", type=str, default="./test_list.txt")
    parser.add_argument("--valid_list_file", type=str, default="./valid_list.txt")
 
    # after your existing parser args:
    parser.add_argument("--z_norm", action="store_true", help="Enable z-normalization of node/edge features based on the training set")
    parser.add_argument("--zstats_path", type=str, default="./checkpoints/znorm.pt", help="Path to save/load z-norm stats (torch.save)")
 
    parser.add_argument("--file_type", type=str, default="pdb")
    parser.add_argument("--valid_num", type=int, default=512)
 
    parser.add_argument("--enable_sadic", action="store_true")
 
    # --- W&B specific flags
    parser.add_argument("--wandb_project", type=str, default="ampnn")
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
 
    if args.sweep_config:
        # Start a new sweep
        with open(args.sweep_config, "r") as f:
            sweep_config = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
        wandb.agent(sweep_id, function=partial(sweep_main, args))
    elif args.sweep_id:
        # Resume an existing sweep
        wandb.agent(args.sweep_id, function=partial(sweep_main, args))
    else:
        # Single run
        run_training(args)