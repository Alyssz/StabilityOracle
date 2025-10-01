# train_masked_residue.py
import os, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from lettore_dati import get_protein_df
from preprocessing_utils import filter_het_atoms
from prova_dataloader import MolExample, MolDataset, pad_collate
from blocks import Backbone

# >>> SWITCH FEAT #2: usa SADIC al posto di SASA
USE_SADIC = False   # False = SASA, True = SADIC

# ============ Config ============
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ Label: prima lettera di res_id -> classe ============
AA_SINGLE_TO_NAME = {
    "A":"Ala","C":"Cys","D":"Asp","E":"Glu","F":"Phe",
    "G":"Gly","H":"His","I":"Ile","K":"Lys","L":"Leu",
    "M":"Met","N":"Asn","P":"Pro","Q":"Gln","R":"Arg",
    "S":"Ser","T":"Thr","V":"Val","W":"Trp","Y":"Tyr",
}
AA_SINGLE_TO_IDX = {aa:i for i, aa in enumerate(AA_SINGLE_TO_NAME.keys())}

def res_id_to_label(res_id: str, return_name=False):
    if not isinstance(res_id, str) or not res_id:
        raise ValueError(f"res_id non valido: {res_id}")
    aa_letter = res_id[0].upper()
    if aa_letter not in AA_SINGLE_TO_NAME:
        raise ValueError(f"Prima lettera '{aa_letter}' non è un AA atteso per {res_id}")
    return AA_SINGLE_TO_NAME[aa_letter] if return_name else AA_SINGLE_TO_IDX[aa_letter]

# ============ Dataset con label ============
class LabeledMolDataset(Dataset):
    def __init__(self, base_ds: Dataset):
        self.base = base_ds
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        item = self.base[idx]          # dict: atom_types, coords, feats(pp), ca, mask, res_id, (sadic)
        rid = item["res_id"]
        item["label"] = torch.tensor(res_id_to_label(rid), dtype=torch.long)
        return item

def pad_collate_with_labels(batch):
    out = pad_collate(batch)
    # compat: pad_collate può restituire 6 o 7 voci
    if isinstance(out, (list, tuple)):
        if len(out) == 7:
            feats, atom_types, coords, ca, mask, sadic, res_ids = out
        elif len(out) == 6:
            feats, atom_types, coords, ca, mask, sadic = out
            res_ids = [b["res_id"] for b in batch]
        else:
            raise ValueError(f"pad_collate ha restituito {len(out)} elementi (attesi 6 o 7).")
    else:
        raise TypeError("pad_collate deve restituire una tupla/lista.")

    labels = torch.stack([b["label"] for b in batch], dim=0)
    return feats, atom_types, coords, ca, mask, sadic, res_ids, labels

# ============ Split ============
def split_indices(n: int, val_frac=0.1, test_frac=0.1, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n); rng.shuffle(idx)
    n_test = int(round(n * test_frac))
    n_val  = int(round(n * val_frac))
    test_idx = idx[:n_test].tolist()
    val_idx  = idx[n_test:n_test+n_val].tolist()
    train_idx = idx[n_test+n_val:].tolist()
    return train_idx, val_idx, test_idx

# ============ Costruzione pp (SASA vs SADIC) ============
def build_pp(feats, sadic, mask, use_sadic: bool):
    """
    feats: (B,L,2) [charge, SASA] dal collate
    Se use_sadic=True -> rimpiazza colonna 1 con SADIC (solo dove mask=True)
    """
    pp = feats.clone().float()
    if use_sadic:
        if sadic is None:
            raise ValueError("use_sadic=True ma 'sadic' non disponibile nel batch.")
        pp[..., 1] = 0.0
        pp[..., 1][mask] = sadic.float()[mask]
    return pp

# ============ Valutazione ============
@torch.no_grad()
def evaluate(model, loader, criterion, use_sadic=False):
    model.eval()
    total_loss, total, correct, top5 = 0.0, 0, 0, 0
    for batch in loader:
        feats, atom_types, coords, ca, mask, sadic, res_ids, labels = batch

        pp = build_pp(feats, sadic, mask, use_sadic).to(device)
        atom_types = atom_types.clamp(min=0).to(device)  # embedding-safe
        coords = coords.float().to(device)
        ca     = ca.float().to(device)
        mask   = mask.bool().to(device)
        labels = labels.to(device)

        _, logits = model([None], atom_types, pp, coords, ca, mask)
        loss = criterion(logits, labels)

        total_loss += float(loss) * labels.size(0)
        total += labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += int((preds == labels).sum())

        k = min(5, logits.shape[-1])
        topk = torch.topk(logits, k=k, dim=-1).indices
        top5 += int((topk == labels.unsqueeze(-1)).any(dim=-1).sum())

    return {"loss": total_loss / max(total,1),
            "acc":  correct / max(total,1),
            "top5": top5  / max(total,1),
            "n":    total}

# ============ Training ============
def train_loop(model, train_loader, val_loader, epochs=20, lr=1e-3, wd=1e-2,
               grad_clip=1.0, amp=True, use_sadic=False):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type=="cuda"))


    best_val = math.inf
    best_state = None

    for epoch in range(1, epochs+1):
        model.train()
        running_loss, seen = 0.0, 0

        for batch in train_loader:

            feats, atom_types, coords, ca, mask, sadic, res_ids, labels = batch

            pp = build_pp(feats, sadic, mask, use_sadic).to(device)
            atom_types = atom_types.clamp(min=0).to(device)
            coords = coords.float().to(device)
            ca     = ca.float().to(device)
            mask   = mask.bool().to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(amp and device.type=="cuda")):
                _, logits = model([None], atom_types, pp, coords, ca, mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            bs = labels.size(0)
            running_loss += float(loss) * bs
            seen += bs

        train_loss = running_loss / max(seen,1)
        val_metrics = evaluate(model, val_loader, criterion, use_sadic=use_sadic)

        print(f"[{epoch:03d}] train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} val_top5={val_metrics['top5']:.3f}")

        # early-stopping su val_loss
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

# ============ Main ============
def main():
    set_seed(42)

    # --- Carica una proteina dall’HDF5 ---
    protein_path = "/mnt/beegfs/home/giulio/transformerSADIC/zernikegrams/test_con_sadic.hdf5"
    df = get_protein_df(protein_path)
    df = filter_het_atoms(df)
    df["atom_name"] = df["atom_name"].astype(str).str.strip().str.upper()

    # --- Residui validi (con CA) ---
    has_ca = (df["atom_name"] == "CA").groupby(df["res_id"]).any()
    valid_res = has_ca[has_ca].index.tolist()
    print(f"[INFO] Residui validi con CA: {len(valid_res)}")
    print(f"[DEBUG] primi 5 res_id: {valid_res[:5]}")

    # --- Crea MolExample per tutti i residui della proteina ---
    mols = []
    for r in valid_res:
        mol = MolExample(df=df, res_id=r)
        mols.append(mol)
        if len(mols) <= 2:
            print(f"\n[MolExample] res_id={r}")
            print(f"  atom_types shape: {mol.atom_types.shape}")
            print(f"  coords shape:     {mol.coords.shape}")
            print(f"  pp shape:         {mol.pp.shape}")
            print(f"  ca_coo:           {mol.ca_coo}")
            print(f"  mask:             {mol.mask}")

    base_ds = MolDataset(mols)
    print(f"\n[MolDataset] len: {len(base_ds)}")
    print(f"  Primo elemento keys: {list(base_ds[0].keys())}")
    print(f"  Primo elemento res_id: {base_ds[0]['res_id']}")

    ds = LabeledMolDataset(base_ds)
    print(f"\n[LabeledMolDataset] len: {len(ds)}")
    print(f"  Primo label: {ds[0]['label']} ({AA_SINGLE_TO_NAME[valid_res[0][0]]})")
    print(f"  Primo res_id: {ds[0]['res_id']}")

    # --- Split ---
    train_idx, val_idx, test_idx = split_indices(len(ds), val_frac=0.1, test_frac=0.1, seed=42)
    train_ds, val_ds, test_ds = Subset(ds, train_idx), Subset(ds, val_idx), Subset(ds, test_idx)
    print(f"[INFO] Dataset: total={len(ds)}  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # --- DataLoader ---
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  collate_fn=pad_collate_with_labels, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, collate_fn=pad_collate_with_labels, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, collate_fn=pad_collate_with_labels, num_workers=0)
    print(f"[INFO] Batches: train={len(train_loader)} val={len(val_loader)} test={len(test_loader)}")
    print(f"[INFO] Feature #2: {'SADIC' if USE_SADIC else 'SASA'}")

    # --- Modello ---
    args = type("Args", (), {"debug": "", "use_sadic": USE_SADIC})()
    model = Backbone(args=args)

    # --- Train+Val ---
    model = train_loop(model, train_loader, val_loader,
                       epochs=20, lr=1e-3, wd=1e-2, grad_clip=1.0, amp=True,
                       use_sadic=USE_SADIC)

    # --- Test ---
    metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), use_sadic=USE_SADIC)
    print(f"[TEST] loss={metrics['loss']:.4f} acc={metrics['acc']:.3f} top5={metrics['top5']:.3f} n={metrics['n']}")

if __name__ == "__main__":
    main()
