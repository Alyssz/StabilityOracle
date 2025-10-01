# data_loader.py
import os
import math
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# ------------------------------
# Mapping elementi → indici (coerente col modello)
# ------------------------------
def map_element_to_index(e: str) -> int:
    e = str(e).strip()
    if not e:
        return 8
    e_up = e.upper()
    if e_up in ("CL", "BR", "I"):
        return 7
    return {"H":0, "C":1, "N":2, "O":3, "F":4, "S":5, "P":6}.get(e_up, 8)  # 8=unknown

# opzionale: mappa resname → label 0..19 se non hai già aa_label nel file
_AA_MAP = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4, "GLU": 5, "GLN": 6, "GLY": 7,
    "HIS": 8, "ILE": 9, "LEU":10, "LYS":11, "MET":12, "PHE":13, "PRO":14, "SER":15,
    "THR":16, "TRP":17, "TYR":18, "VAL":19
}

# ------------------------------
# Dataset HDF5 (via pandas.HDFStore)
# ------------------------------
class MicroenvH5Dataset(Dataset):
    """
    Richiede nel file HDF5 una tabella 'microenv_pairs' (merged con features per atomo).
    Colonne minime attese:
      - center_* : center_res_id, center_chain_id, center_resname, center_ca_x/y/z
      - nbr_*    : nbr_res_id, nbr_atom_name, nbr_x/y/z, distance
      - features : element, charge, e UNA tra {SASA, SADIC} (puoi rinominarla via args)
      - (label)  : opzionale 'aa_label' per il centro; altrimenti si usa center_resname -> _AA_MAP

    Ogni __getitem__ restituisce UN centro con i suoi vicini:
      atom_types: (n,) int64  [0..8]
      pp:         (n,2) float32  [charge, sasa_or_sadic]
      coords:     (n,3) float32
      ca:         (3,)  float32
      mask:       (n,)  float32  (1=valido)
      label:      ()    int64    [0..19]
    """
    def __init__(
        self,
        h5_path: str,
        pairs_key: str = "microenv_pairs",
        element_col: str = "element",
        charge_col: str = "charge",
        pp2_col: str = "SASA",            # oppure "SADIC"
        use_sadic: bool = False,          # se True, forza pp2_col="SADIC" se presente
        exclude_h: bool = False,
        max_neighbors: int | None = 512,  # None = nessun limite
        max_distance: float | None = None, # se set, filtra vicini oltre questa distanza
        require_label_col: bool = False,   # se True, obbliga presenza 'aa_label'
        cache_in_memory: bool = False,     # carica tutto in RAM (veloce ma usa memoria)
    ):
        super().__init__()
        self.h5_path = h5_path
        self.pairs_key = pairs_key
        self.element_col = element_col
        self.charge_col = charge_col
        self.pp2_col = "SADIC" if use_sadic else pp2_col
        self.exclude_h = exclude_h
        self.max_neighbors = max_neighbors
        self.max_distance = max_distance
        self.require_label_col = require_label_col
        self.cache_in_memory = cache_in_memory

        # Leggi tabella
        with pd.HDFStore(self.h5_path, mode="r") as store:
            if self.pairs_key not in store:
                raise KeyError(f"Key '{self.pairs_key}' non trovata in {self.h5_path}")
            df = store[self.pairs_key]

        # Validazione colonne minime
        needed = {
            "center_res_id","center_chain_id","center_resname","center_ca_x","center_ca_y","center_ca_z",
            "nbr_res_id","nbr_atom_name","nbr_x","nbr_y","nbr_z","distance",
            self.element_col, self.charge_col
        }
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Mancano colonne in '{self.pairs_key}': {missing}")
        if self.pp2_col not in df.columns:
            if self.require_label_col:  # messaggio più chiaro se stai forzando schema
                raise ValueError(f"Colonna '{self.pp2_col}' non trovata in '{self.pairs_key}'")
            # Se non c'è SASA/SADIC, crea colonna zeros
            df[self.pp2_col] = 0.0

        # Filtri opzionali
        if self.max_distance is not None:
            df = df[df["distance"] <= float(self.max_distance)].copy()
        if self.exclude_h:
            # Filtra idrogeni sia per nome atomo che per elemento
            mask_h = (
                df["nbr_atom_name"].astype(str).str.fullmatch(r"\s*H\d*") |
                (df[self.element_col].astype(str).str.upper() == "H")
            )
            df = df[~mask_h].copy()

        # Determina etichette (aa_label)
        if "aa_label" not in df.columns:
            if self.require_label_col:
                raise ValueError("Manca 'aa_label' e require_label_col=True")
            # fallback da center_resname
            lab = df["center_resname"].astype(str).str.upper().map(_AA_MAP).astype("Int64")
            if lab.isna().any():
                # residui non standard → droppali
                df = df[~lab.isna()].copy()
                lab = df["center_resname"].astype(str).str.upper().map(_AA_MAP).astype(int)
            df["aa_label"] = lab.astype(int)

        # Lista centri (unici)
        centers = (
            df[["center_res_id","center_chain_id","center_resname","center_ca_x","center_ca_y","center_ca_z","aa_label"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # opzionale caching per velocità
        self._df = df if not cache_in_memory else df.copy()
        self._centers = centers if not cache_in_memory else centers.copy()

        # build indice booleano per ogni centro → posizioni nel df (più veloce con groupby)
        self._groups = self._df.groupby("center_res_id").indices

    def __len__(self) -> int:
        return len(self._centers)

    def _rows_for_center(self, center_res_id: str) -> pd.DataFrame:
        # usa l’indice precalcolato (veloce)
        idx = self._groups.get(center_res_id, None)
        if idx is None or len(idx) == 0:
            return pd.DataFrame(columns=self._df.columns)
        return self._df.iloc[idx]

    def __getitem__(self, idx: int):
        c = self._centers.iloc[idx]
        center_id = c["center_res_id"]
        ca = np.array([c["center_ca_x"], c["center_ca_y"], c["center_ca_z"]], dtype=np.float32)
        label = int(c["aa_label"])

        neigh = self._rows_for_center(center_id).copy()
        if neigh.empty:
            # nessun vicino: crea un dummy con N=1 (tutto zero) per evitare crash; label resta valida
            atom_types = torch.zeros(1, dtype=torch.long)
            pp = torch.zeros(1, 2, dtype=torch.float32)
            coords = torch.zeros(1, 3, dtype=torch.float32)
            mask = torch.zeros(1, dtype=torch.float32)
            return {
                "atom_types": atom_types, "pp": pp, "coords": coords,
                "ca": torch.from_numpy(ca), "mask": mask, "label": torch.tensor(label, dtype=torch.long),
                "center_res_id": center_id,
            }

        # Limita numero vicini (se richiesto) scegliendo i più vicini al Cα
        if self.max_neighbors is not None and len(neigh) > self.max_neighbors:
            neigh = neigh.nsmallest(self.max_neighbors, "distance")

        # Tensori
        coords = neigh[["nbr_x","nbr_y","nbr_z"]].to_numpy(dtype=np.float32)          # (n,3)
        elements = neigh[self.element_col].astype(str).map(map_element_to_index).fillna(8).to_numpy(np.int64)
        charge = neigh[self.charge_col].to_numpy(dtype=np.float32) if self.charge_col in neigh.columns else np.zeros(len(neigh), np.float32)
        pp2 = neigh[self.pp2_col].to_numpy(dtype=np.float32) if self.pp2_col in neigh.columns else np.zeros(len(neigh), np.float32)
        pp = np.stack([charge, pp2], axis=-1)                                         # (n,2)

        item = {
            "atom_types": torch.as_tensor(elements, dtype=torch.long),
            "pp": torch.as_tensor(pp, dtype=torch.float32),
            "coords": torch.as_tensor(coords, dtype=torch.float32),
            "ca": torch.as_tensor(ca, dtype=torch.float32),
            "mask": torch.ones(len(neigh), dtype=torch.float32),  # 1=valido
            "label": torch.tensor(label, dtype=torch.long),
            "center_res_id": center_id,
        }
        return item

# ------------------------------
# Collate con padding
# ------------------------------
def pad_collate(batch):
    """
    Ritorna tensori paddati a Nmax:
      atom_types (B,N), pp (B,N,2), coords (B,N,3), ca (B,3), mask (B,N), label (B,)
    """
    B = len(batch)
    Nmax = max(x["atom_types"].shape[0] for x in batch) if B > 0 else 1

    def pad1d(t: torch.Tensor, N: int, fill=0):
        out = t.new_full((N,), fill)
        out[: t.shape[0]] = t
        return out

    def pad2d(t: torch.Tensor, N: int, D: int, fill=0.0):
        out = t.new_full((N, D), fill)
        out[: t.shape[0], :] = t
        return out

    atom_types = torch.stack([pad1d(b["atom_types"], Nmax, fill=0) for b in batch], dim=0)
    pp = torch.stack([pad2d(b["pp"], Nmax, 2, fill=0.0) for b in batch], dim=0)
    coords = torch.stack([pad2d(b["coords"], Nmax, 3, fill=0.0) for b in batch], dim=0)
    ca = torch.stack([b["ca"] for b in batch], dim=0)  # (B,3)
    mask = torch.stack([pad1d(b["mask"], Nmax, fill=0.0) for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)

    # utile se vuoi tracciare il centro durante il debug
    centers = [b.get("center_res_id", None) for b in batch]

    return dict(atom_types=atom_types, pp=pp, coords=coords, ca=ca, mask=mask, label=labels, centers=centers)

# ------------------------------
# Split per centro + DataLoader
# ------------------------------
def split_indices_by_center(centers: pd.Series, train=0.8, val=0.1, test=0.1, seed=42):
    """
    Split deterministico per center_res_id (no leakage tra split).
    """
    assert abs(train + val + test - 1.0) < 1e-6
    uniq = centers.drop_duplicates().tolist()
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(n * train))
    n_va = int(round(n * val))
    tr_ids = set(uniq[:n_tr])
    va_ids = set(uniq[n_tr:n_tr+n_va])
    te_ids = set(uniq[n_tr+n_va:])
    # mappa a indici dataset
    idx_tr, idx_va, idx_te = [], [], []
    for i, cid in enumerate(centers.tolist()):
        if cid in tr_ids: idx_tr.append(i)
        elif cid in va_ids: idx_va.append(i)
        else: idx_te.append(i)
    return idx_tr, idx_va, idx_te

def get_loaders(
    h5_path: str,
    pairs_key: str = "microenv_pairs",
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    use_sadic: bool = False,
    exclude_h: bool = False,
    max_neighbors: int | None = 512,
    max_distance: float | None = None,
    require_label_col: bool = False,
    cache_in_memory: bool = False,
    seed: int = 42,
    splits=(0.8, 0.1, 0.1),
):
    """
    Crea dataset + dataloader per train/val/test.
    """
    ds = MicroenvH5Dataset(
        h5_path=h5_path,
        pairs_key=pairs_key,
        use_sadic=use_sadic,
        exclude_h=exclude_h,
        max_neighbors=max_neighbors,
        max_distance=max_distance,
        require_label_col=require_label_col,
        cache_in_memory=cache_in_memory,
    )

    centers_series = ds._centers["center_res_id"]
    idx_tr, idx_va, idx_te = split_indices_by_center(
        centers_series, train=splits[0], val=splits[1], test=splits[2], seed=seed
    )

    ds_tr = Subset(ds, idx_tr)
    ds_va = Subset(ds, idx_va)
    ds_te = Subset(ds, idx_te)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_memory,
                       persistent_workers=persistent_workers, collate_fn=pad_collate)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                       num_workers=max(1, num_workers//2), pin_memory=pin_memory,
                       persistent_workers=persistent_workers, collate_fn=pad_collate)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                       num_workers=max(1, num_workers//2), pin_memory=pin_memory,
                       persistent_workers=persistent_workers, collate_fn=pad_collate)
    return ds, (ds_tr, ds_va, ds_te), (dl_tr, dl_va, dl_te)
