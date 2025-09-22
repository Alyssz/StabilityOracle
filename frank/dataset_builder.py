"""
Preprocessing degli HDF5 → DataFrame pulito + indici residui.

✅ Versione semplificata: niente split validation, tutti i protein_id sono training.

Output (in --out_dir):
  - atoms.parquet : tabella atomica pulita (una riga per atomo)
  - residues_with_CA.csv : elenco (protein_id,res_id) che hanno almeno un atomo CA
  - proteins_train.txt   : lista protein_id inclusi (tutti training)
  - PREPROC_SUMMARY.txt  : report riassuntivo

Uso (solo proteine in keep.txt):
  python preprocess_hdf5.py --h5 path/to/data.h5 --out_dir out/ --pdb_list keep.txt --seed 42

Uso (primi N protein_id):
  python preprocess_hdf5.py --h5 path/to/data.h5 --out_dir out/ --num_proteins 500 --seed 42
"""
from __future__ import annotations
import argparse
import os
from typing import List, Tuple, Optional, Set

import numpy as np
import pandas as pd

import hdf5plugin  # se il file è compresso, deve essere importato prima di h5py
import h5py

# ---------------------
# Utilities for parsing
# ---------------------

def _dec(x):
    return x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else str(x)


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["atom_name","element","res_id"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
    out["atom_name"] = (out["atom_name"].str.upper()
                        .str.replace("\u03b1","A",regex=False)
                        .str.replace("\u0391","A",regex=False)
                        .str.replace(" ","",regex=False))
    out["element"] = out["element"].str.upper()
    return out


def filter_het_atoms(protein_df: pd.DataFrame) -> pd.DataFrame:
    structure = protein_df.copy()
    return structure[~structure['res_id'].str.startswith('Z')]


def _res_id_per_atom(res_ids_arr) -> np.ndarray:
    out = []
    for row in res_ids_arr:  # (n_atoms, 6)
        parts = [_dec(x).strip() for x in row[:4]]
        out.append("".join(parts) if all(parts) else "")
    return np.array(out, dtype=object)


# ---------------------
# Core preprocessing
# ---------------------

def load_all_proteins_as_df(
    h5_path: str,
    max_entries: Optional[int] = None,
    include_set: Optional[Set[str]] = None,
) -> pd.DataFrame:
    rows = []
    seen: Set[str] = set()
    want = {pid.upper() for pid in include_set} if include_set else None

    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            raise ValueError("Nel file HDF5 non c'è il dataset '/data'.")
        dset = f["data"]
        if not getattr(dset, "dtype", None) or not dset.dtype.fields:
            raise ValueError("'/data' non è un dataset strutturato.")

        N = dset.shape[0]
        if want is not None:
            print(f"[INFO] Scansione di {N} entries; target lista = {len(want)} protein_id")
        else:
            M = N if max_entries is None else min(max_entries, N)
            print(f"[INFO] Carico {M}/{N} proteine da /data …")

        for i in range(N):
            # Se ho una lista, continuo finché non ho trovato tutti
            if want is not None and seen.issuperset(want):
                break
            # Se NON ho lista ma un limite, fermati a max_entries diversi
            if want is None and max_entries is not None and len(seen) >= max_entries:
                break

            e = dset[i]
            pdb_code = _dec(e["pdb"]).strip().upper()

            # Se ho la lista e questo pdb non c'è, salto
            if want is not None and pdb_code not in want:
                continue

            atom_names = e["atom_names"]
            elements   = e["elements"]
            coords     = e["coords"]
            sasas      = e["SASA"]
            charges    = e["charges"]
            sadic      = e["SADIC"]
            res_ids    = e["res_ids"]

            atom_names_str = atom_names.astype(str)
            valid_mask = np.char.strip(atom_names_str) != ""
            if not valid_mask.any():
                continue
            res_id_flat = _res_id_per_atom(res_ids)[valid_mask]

            df_i = pd.DataFrame({
                "atom_name": atom_names_str[valid_mask],
                "element":   elements[valid_mask].astype(str),
                "x":         coords[valid_mask, 0],
                "y":         coords[valid_mask, 1],
                "z":         coords[valid_mask, 2],
                "SASA":      sasas[valid_mask],
                "charge":    charges[valid_mask],
                "res_id":    res_id_flat,
                "sadic":     sadic[valid_mask],
                "protein_id": pdb_code,
            })
            df_i = _normalize_df(df_i)
            df_i = filter_het_atoms(df_i)
            rows.append(df_i)
            seen.add(pdb_code)

        if want is not None:
            missing = sorted(want - seen)
            if missing:
                print(f"[WARN] {len(missing)} ID della lista non trovati nell'HDF5. Esempi: {missing[:10]}")
                # opzionale: salva i mancanti
                miss_path = os.path.join(os.path.dirname(h5_path), "missing_pdb_from_list.txt")
                try:
                    with open(miss_path, "w") as mf:
                        mf.write("\n".join(missing))
                    print(f"[INFO] Salvata lista mancanti: {miss_path}")
                except Exception as _:
                    pass

        if not rows:
            raise ValueError("Nessuna entry valida trovata in '/data' per i criteri selezionati.")
        big = pd.concat(rows, ignore_index=True)
        print(f"[INFO] DF finale: shape={big.shape} | proteine={big['protein_id'].nunique()}")
        return big

_AA20 = set(list("ACDEFGHIKLMNPQRSTVWY"))

def derive_aa_one_from_resid(res_id: str) -> str:
    """
    Estrae la prima lettera di res_id e la normalizza a maiuscolo.
    Se non è tra le 20 standard, ritorna 'X' (oppure gestisci come preferisci).
    """
    if not isinstance(res_id, str) or len(res_id) == 0:
        return "X"
    aa = res_id[0].upper()
    return aa if aa in _AA20 else "X"


def compute_residue_index_with_CA(df_all: pd.DataFrame) -> pd.DataFrame:
    has_ca = (df_all["atom_name"].str.upper() == "CA").groupby(
        [df_all["protein_id"], df_all["res_id"]]
    ).any()
    res_df = has_ca[has_ca].reset_index()[["protein_id","res_id"]]
    # aggiungi la label aa_one
    res_df["aa_one"] = res_df["res_id"].astype(str).map(derive_aa_one_from_resid)
    return res_df



def write_summary(out_dir: str, df_all: pd.DataFrame, res_df: pd.DataFrame):
    lines = []
    lines.append(f"Atoms table: {df_all.shape[0]} righe, {df_all['protein_id'].nunique()} proteine")
    lines.append(f"Residui con CA: {len(res_df)} (unique residui: {res_df[['protein_id','res_id']].drop_duplicates().shape[0]})")
    aa = df_all['atom_name'].value_counts().head(10)
    lines.append("Top10 atom_name:\n" + aa.to_string())
    with open(os.path.join(out_dir, 'PREPROC_SUMMARY.txt'), 'w') as f:
        f.write("\n".join(lines))


# ---------------------
# CLI
# ---------------------

def _read_pdb_list(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Lista PDB non trovata: {path}")
    with open(path, 'r') as f:
        ids = [line.strip() for line in f if line.strip() and not line.lstrip().startswith('#')]
    ids_norm = [s.upper() for s in ids]
    print(f"[INFO] Lista PDB: {len(ids)} righe, {len(set(ids_norm))} unici dopo normalizzazione")
    return set(ids_norm)

"""
training_h5_path   = "/mnt/beegfs/home/giulio/transformerSADIC/reset/data/training/training_structural_info_with_SADIC.hdf5"
validation_h5_path = "/mnt/beegfs/home/giulio/transformerSADIC/reset/data/validation/validation_structural_info_with_SADIC.hdf5"
test_h5_path       = "/mnt/beegfs/home/giulio/transformerSADIC/reset/data/test/test_structural_info_with_SADIC.hdf5"

"""
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True, help='Path HDF5 con /data')
    ap.add_argument('--out_dir', required=True, help='Cartella di output')
    ap.add_argument('--num_proteins', type=int, default=None,
                    help='Usa solo se NON passi --pdb_list: prende i primi N protein_id presenti nel file')
    ap.add_argument('--pdb_list', type=str, default=None,
                    help='File .txt con i protein_id da includere (uno per riga); se presente, ignora --num_proteins')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    include = _read_pdb_list(args.pdb_list)

    # ⚠️ Se c'è la lista, ignoriamo num_proteins (ed avvisiamo)
    if include is not None and args.num_proteins is not None:
        print(f"[WARN] Hai passato sia --pdb_list che --num_proteins={args.num_proteins}. "
              f"Ignoro --num_proteins e uso SOLO la lista.")
        args.num_proteins = None

    df_all = load_all_proteins_as_df(
        args.h5,
        max_entries=args.num_proteins,   # sarà None se include != None
        include_set=include,
    )

    res_df = compute_residue_index_with_CA(df_all)

    atoms_path = os.path.join(args.out_dir, 'atoms.parquet')
    residx_path = os.path.join(args.out_dir, 'residues_with_CA.csv')
    train_path = os.path.join(args.out_dir, 'proteins_train.txt')

    df_all.to_parquet(atoms_path, index=False)
    res_df.to_csv(residx_path, index=False)
    with open(train_path, 'w') as f:
        f.write("\n".join(sorted(df_all['protein_id'].drop_duplicates())))

    write_summary(args.out_dir, df_all, res_df)

    print(f"[OK] Salvato: {atoms_path}\n[OK] Salvato: {residx_path}\n[OK] Salvato: {train_path}")

if __name__ == '__main__':
    main()
