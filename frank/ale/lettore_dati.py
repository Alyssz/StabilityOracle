import hdf5plugin  # <--- IMPORTA PRIMA DI h5py
import h5py
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser, is_aa


# def esplora_hdf5(percorso_file):
#     with h5py.File(percorso_file, "r") as f:
#         def visita(nome, oggetto):
#             print(f"📂 {nome}")
#             if isinstance(oggetto, h5py.Dataset):
#                 print(f"   └─ Dataset → shape: {oggetto.shape}, dtype: {oggetto.dtype}")
#         f.visititems(visita)

def get_protein_df(path_hdf5: str):
    # esplora_hdf5(path_hdf5)

    with h5py.File(path_hdf5, "r") as f:
        dset = f["data"]
        entry = dset[:1].copy()
        print(50*"#")
        print("Extracting Protein ...\n")
        print("Compressione:", dset.compression)
        print("Shape:", dset.shape)
        print("PDB code:", entry["pdb"][0].decode())
        print(dset)
        print(entry)
        pdb_code = entry['pdb'][0].decode()
        atom_names = entry['atom_names']
        elements = entry['elements']
        res_ids = entry['res_ids']
        coords = entry['coords']
        sasas = entry['SASAs']
        charges = entry['charges']
        sadic = entry['sadic']

        # Pulisci i dati rimuovendo zeri finali
        valid_mask = np.char.strip(atom_names.astype(str)) != ""
        n_valid = np.sum(valid_mask)

        print(f"\nPDB code: {pdb_code}, Valid Atoms: {n_valid}")
        res_id_strs = []
        for i in range(len(res_ids)):
            current_res = res_ids[i]  # è una matrice (n_atoms, 6)
            row_strs = []

            for row in current_res:
                decoded = [x.decode('utf-8').strip() for x in row[:4]]  # Prendi solo i primi 4 campi
                if all(decoded):  # scarta righe vuote
                    row_strs.append("".join(decoded))

            res_id_strs.append(row_strs)

        res_id_flat = [item for sublist in res_id_strs for item in sublist]

        df = pd.DataFrame({
            "atom_name": atom_names[valid_mask].astype(str),
            "element": elements[valid_mask].astype(str),
            "x": coords[valid_mask, 0],
            "y": coords[valid_mask, 1],
            "z": coords[valid_mask, 2],
            "SASA": sasas[valid_mask],
            "charge": charges[valid_mask],
            "res_id": res_id_flat,
            "sadic": sadic[valid_mask]
        })
        print("\nNumber of residues: ", df['res_id'].unique().shape[0])
        print(f"Protein {pdb_code} loaded with {len(df)} atoms.")

        sadic_vals = df["sadic"].to_numpy()
        print(f"SADIC min={sadic_vals.min():.4f}  max={sadic_vals.max():.4f}")

        print(df)
        return df


def get_protein_df_from_pdb(
    path_pdb: str,
    model_index: int = 0,
    chain_id: str | None = None,
    altloc_priority: str = "A",
    sasa_from: str = "bfactor",  # "bfactor" or "occupancy"
    sadic_from: str = "occupancy",  # "occupancy" or "bfactor"
):
    """
    Load a protein from a PDB/mmCIF file into a pandas DataFrame, reading:
      - SASA from the chosen per-atom column (default: B-factor)
      - SADIC from the chosen per-atom column (default: occupancy)

    Returns columns: atom_name, element, x, y, z, SASA, charge, res_id, sadic
    """
    # Pick parser by extension
    ext = path_pdb.lower().split(".")[-1]
    parser = MMCIFParser(QUIET=True) if ext in {"cif", "mmcif"} else PDBParser(QUIET=True)
    structure = parser.get_structure("protein", path_pdb)

    # Select model
    try:
        model = list(structure)[model_index]
    except IndexError:
        raise ValueError(f"Model index {model_index} not found in file.")

    def _keep_altloc(atom):
        altloc = atom.get_altloc()
        return (altloc == " ") or (altloc == altloc_priority)

    def _get_val(atom, source: str):
        # Map requested source to Biopython getters
        if source == "bfactor":
            v = atom.get_bfactor()
        elif source == "occupancy":
            v = atom.get_occupancy()
        else:
            raise ValueError("source must be 'bfactor' or 'occupancy'")
        # Biopython may return None for occupancy; coerce to NaN
        return float(v) if v is not None else np.nan

    records = []
    pdb_code = getattr(structure, "id", "UNKNOWN")

    for ch in model:
        if chain_id is not None and ch.id != chain_id:
            continue

        for res in ch:
            hetflag, resseq, icode = res.id
            if not is_aa(res, standard=True):
                continue

            resname = res.get_resname().strip()
            resid_compact = f"{resname}{resseq}{(icode or '').strip()}:{ch.id}"

            for atom in res.get_atoms():
                if not _keep_altloc(atom):
                    continue

                atom_name = atom.get_name().strip()
                element = (atom.element or "").strip()
                if not element:
                    element = "".join([c for c in atom_name if c.isalpha()])[:2].title()

                x, y, z = atom.get_coord()
                sasa = _get_val(atom, sasa_from)
                sadic = _get_val(atom, sadic_from)

                records.append(
                    {
                        "atom_name": atom_name,
                        "element": element,
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "SASA": sasa,          # from B-factor by default
                        "charge": np.nan,      # not stored in standard PDB/mmCIF
                        "res_id": resid_compact,
                        "sadic": sadic,        # from occupancy by default
                    }
                )

    if not records:
        raise ValueError("No protein atoms found with the given filters.")

    df = pd.DataFrame.from_records(records)

    # Console summary (mirrors your style)
    print(50 * "#")
    print("Extracting Protein from PDB/mmCIF ...\n")
    print(f"PDB code: {pdb_code}")
    if chain_id:
        print(f"Chain: {chain_id}")
    print(f"Model: {model_index}")
    print(f"Atoms: {len(df)}")
    print("Number of residues:", df["res_id"].nunique())

    sasa_vals = df["SASA"].to_numpy()
    if np.isfinite(sasa_vals).any():
        print(f"SASA min={np.nanmin(sasa_vals):.4f}  max={np.nanmax(sasa_vals):.4f}")

    sadic_vals = df["sadic"].to_numpy()
    if np.isfinite(sadic_vals).any():
        print(f"SADIC min={np.nanmin(sadic_vals):.4f}  max={np.nanmax(sadic_vals):.4f}")

    return df


if __name__ == "__main__":
    print(get_protein_df('/mnt/beegfs/home/giulio/transformerSADIC/reset/data/training/training_structural_info_with_SADIC.hdf5'))