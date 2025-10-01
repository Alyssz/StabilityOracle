import numpy as np
import torch
import pandas as pd

# --- mapping elementi -> id (maiuscoli e senza spazi) ---
ATOM_TYPES = {
    "H": 0, "C": 1, "N": 2, "O": 3, "F": 4, "S": 5, "P": 6,
    "CL": 7, "BR": 7, "I": 7,   # alogeni raggruppati
}

AA_SINGLE_TO_NAME = {
    "A":"Ala","C":"Cys","D":"Asp","E":"Glu","F":"Phe",
    "G":"Gly","H":"His","I":"Ile","K":"Lys","L":"Leu",
    "M":"Met","N":"Asn","P":"Pro","Q":"Gln","R":"Arg",
    "S":"Ser","T":"Thr","V":"Val","W":"Trp","Y":"Tyr",
}
# ✅ MANCAVA
AA_SINGLE_TO_IDX = {aa: i for i, aa in enumerate(AA_SINGLE_TO_NAME.keys())}

def res_id_to_label(res_id: str, return_name=False):
    if not res_id or not isinstance(res_id, str):
        raise ValueError(f"res_id non valido: {res_id}")
    aa_letter = res_id[0].upper()
    if aa_letter not in AA_SINGLE_TO_NAME:
        raise ValueError(f"Codice AA '{aa_letter}' non riconosciuto per {res_id}")
    return AA_SINGLE_TO_NAME[aa_letter] if return_name else AA_SINGLE_TO_IDX[aa_letter]

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["atom_name", "element", "res_id"]:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda s: s.decode("utf-8", "ignore") if isinstance(s, (bytes, bytearray)) else s
            )
            out[col] = out[col].astype(str).str.strip()
    out["atom_name"] = (out["atom_name"].str.upper()
                        .str.replace("\u03b1","A", regex=False)  # α -> A
                        .str.replace("\u0391","A", regex=False)  # Α -> A
                        .str.replace(" ","", regex=False))
    out["element"] = out["element"].str.upper()
    return out

def _element_to_id(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip().str.upper()
    return s.map(ATOM_TYPES).fillna(-1).astype(int).to_numpy()  # -1 = padding/ignora

def _get_ca_coord(sub: pd.DataFrame) -> torch.Tensor:
    ca = sub[sub["atom_name"] == "CA"]
    if ca.empty:
        raise ValueError("Nessun Cα trovato per questo residuo.")
    ca_xyz = ca[["x","y","z"]].iloc[0].to_numpy(dtype=np.float32)
    return torch.from_numpy(ca_xyz)[None, :]  # (1,3)

def collate_protein(df: pd.DataFrame, res_id: str, use_sadic: bool = False):
    """
    Raccoglie tensori per un residuo: tipi, coord, feature (charge + SASA/SADIC), Cα, mask, sadic_vec
    """
    sub = df[df["res_id"] == res_id].copy()
    if sub.empty:
        raise ValueError(f"Residuo {res_id} non trovato nel DataFrame.")

    # --- atom types da 'element' (NON da atom_name)
    atom_types_np = _element_to_id(sub["element"])
    atom_types = torch.from_numpy(atom_types_np).long()  # (n,)

    # --- coords
    coords = torch.tensor(sub[["x","y","z"]].to_numpy(dtype=np.float32))  # (n,3)

    # --- feature atomiche: charge + (SASA oppure SADIC)
    charges = torch.tensor(sub["charge"].to_numpy(dtype=np.float32))         # (n,)
    sasa_v  = torch.tensor(sub["SASA"].to_numpy(dtype=np.float32))           # (n,)
    sadic_v = torch.tensor(sub["sadic"].to_numpy(dtype=np.float32))          # (n,)
    second  = sadic_v if use_sadic else sasa_v
    pp = torch.stack([charges, second], dim=1)                                # (n,2)

    # --- Cα
    ca_coo = _get_ca_coord(sub)  # (1,3)

    # --- mask (tutti reali qui; pad arriva dopo)
    mask = torch.ones(len(atom_types), dtype=torch.bool)

    # ✅ ritorniamo anche il vettore SADIC per poterlo usare nel collate (se vuoi switch runtime)
    return atom_types, coords, pp, ca_coo, mask, sadic_v

class MolExample:
    def __init__(self, df: pd.DataFrame, res_id: str, use_sadic: bool = False):
        self.res_id = res_id
        (self.atom_types,
         self.coords,
         self.pp,
         self.ca_coo,
         self.mask,
         self.sadic) = collate_protein(df, res_id, use_sadic=use_sadic)

class MolDataset(torch.utils.data.Dataset):
    def __init__(self, mols):
        self.mols = mols
    def __len__(self): return len(self.mols)
    def __getitem__(self, idx):
        m = self.mols[idx]
        return {
            "atom_types": m.atom_types,    # (n,)
            "sasa":       m.pp[:, 1],      # (n,) (⚠️ se use_sadic=True questa è SADIC rinominata)
            "charge":     m.pp[:, 0],      # (n,)
            "sadic":      m.sadic,         # (n,)
            "coords":     m.coords,        # (n,3)
            "ca":         m.ca_coo,        # (1,3)
            "mask":       m.mask,          # (n,)
            "res_id":     m.res_id,
        }

def pad_collate(batch):
    """
    Ritorna:
      feats(B,L,2), atom_types(B,L), coords(B,L,3), ca(B,1,3),
      mask(B,L), sadic(B,L), res_ids(list)
    """
    B = len(batch)
    lens = [b["atom_types"].shape[0] for b in batch]
    max_len = max(lens)

    def pad_1d(t, fill=0, dtype=None):
        if t is None: return None
        if dtype is None: dtype = t.dtype
        out = t.new_full((max_len,), fill, dtype=dtype)
        out[:t.shape[0]] = t
        return out

    def pad_2d(t, fill=0.0, last_dim=3):
        out = t.new_full((max_len, last_dim), fill, dtype=t.dtype)
        out[:t.shape[0], :] = t
        return out

    atom_types = torch.stack([pad_1d(b["atom_types"], fill=-1, dtype=torch.long) for b in batch], dim=0)  # (B,L)
    coords     = torch.stack([pad_2d(b["coords"].float(), fill=0.0, last_dim=3) for b in batch], dim=0)   # (B,L,3)

    charge = torch.stack([pad_1d(b["charge"].float(), fill=0.0, dtype=torch.float32) for b in batch], dim=0)  # (B,L)
    sasa   = torch.stack([pad_1d(b["sasa"].float(),   fill=0.0, dtype=torch.float32) for b in batch], dim=0)  # (B,L)
    feats  = torch.stack([torch.stack([charge[i], sasa[i]], dim=1) for i in range(B)], dim=0)                 # (B,L,2)

    # sadic
    sadic = torch.stack([pad_1d(b["sadic"].float(), fill=0.0, dtype=torch.float32) for b in batch], dim=0)     # (B,L)

    ca = torch.stack([b["ca"].float() for b in batch], dim=0)  # (B,1,3)
    res_ids = [b["res_id"] for b in batch]

    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, L in enumerate(lens):
        mask[i, :L] = True

    return feats, atom_types, coords, ca, mask, sadic, res_ids
