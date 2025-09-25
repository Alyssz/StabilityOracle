import sys
from dataset_builder import load_all_proteins_as_df

import numpy as np
from tqdm import tqdm
import pandas as pd
import torch

AA_3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

_amino_acids = lambda x: {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLU": 5,
    "GLN": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}.get(x, 20)

# Fixed element mapping for consistent atom type encoding
_element_mapping = {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "F": 4,
    "S": 5,
    "P": 6,
    "Cl": 7,
    "CL": 7,
    "Br": 7,
    "BR": 7,
    "I": 7,
}

AA_1toidx = {
    'A': 0,
    'R': 1,
    'N': 2,
    'D': 3,
    'C': 4,
    'E': 5,
    'Q': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'L': 10,
    'K': 11,
    'M': 12,
    'F': 13,
    'P': 14,
    'S': 15,
    'T': 16,
    'W': 17,
    'Y': 18,
    'V': 19,
}

class GraphTransformerDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_file, pad_to=512, dist_threshold=10.0, max_entries=0, tranform=None):
        self.df = load_all_proteins_as_df(hdf5_file, max_entries=max_entries if max_entries > 0 else None)
        self.examples = None
        self.generate_examples()
        self.pad_to = pad_to
        self.dist_threshold = dist_threshold
        self.transform = tranform

    def __len__(self):
        if self.examples is None:
            raise ValueError("Examples not generated. Call generate_examples() first.")
        return len(self.examples)

    def __getitem__(self, idx):
        if self.examples is None:
            raise ValueError("Examples not generated. Call generate_examples() first.")
        example = self.examples.iloc[idx]
        graph = self.build_graph(example['pdb_id'], example['res_id'], pad_to=self.pad_to, dist_threshold=self.dist_threshold)

        
        return graph, example['label']

    def generate_examples(self):
        examples = []
        for pdb_id in tqdm(self.df['protein_id'].unique()):
            df_pdb = self.df[self.df['protein_id'] == pdb_id]
            for res_id in df_pdb['res_id'].unique():
                if 'CA' not in df_pdb[df_pdb['res_id'] == res_id]['atom_name'].values:
                    continue

                # Clean res_id for examples
                res_id_clean = AA_1toidx.get(res_id[0], 20)

                examples.append({"pdb_id": pdb_id, "res_id": res_id, "label": res_id_clean})

        self.examples = pd.DataFrame(examples)

    def build_graph(self, pdb_id, res_id, pad_to=512, dist_threshold=10.0):
        df_pdb = self.df[self.df['protein_id'] == pdb_id]

        # Skip if CA is missing in residue
        if 'CA' not in df_pdb[df_pdb['res_id'] == res_id]['atom_name'].values:
            return None

        df_res = df_pdb[df_pdb['res_id'] == res_id]
        df_no_res = df_pdb[df_pdb['res_id'] != res_id]

        ca = df_res[df_res['atom_name'] == 'CA'][['x', 'y', 'z']].values[0]

        all_coords = df_no_res[['x', 'y', 'z']].values

        dists = np.linalg.norm(all_coords - ca, axis=1)
        nearby_atoms = df_no_res[dists < dist_threshold]

        graph = {}
        graph['pdb_id'] = pdb_id
        graph['res_id'] = res_id
        num_atoms = len(nearby_atoms)
        mask = np.zeros(pad_to, dtype=np.float32)
        mask[:num_atoms] = 1

        coords = np.zeros((pad_to, 3), dtype=np.float32)
        charges = np.zeros(pad_to, dtype=np.float32)
        sadic = np.zeros(pad_to, dtype=np.float32)
        atom_types = np.zeros(pad_to, dtype=nearby_atoms['element'].dtype)
        sasa = np.zeros(pad_to, dtype=np.float32)

        coords[:num_atoms] = nearby_atoms[['x', 'y', 'z']].values
        charges[:num_atoms] = nearby_atoms['charge'].values
        sadic[:num_atoms] = nearby_atoms['sadic'].values
        atom_types[:num_atoms] = nearby_atoms['element'].values
        sasa[:num_atoms] = nearby_atoms['SASA'].values

        graph['coords'] = coords
        graph['charges'] = charges
        graph['sadic'] = sadic
        graph['atom_types'] = atom_types
        graph['sasa'] = sasa
        graph['mask'] = mask
        graph['ca'] = ca

        return graph

    def collate_fn(self, batch):
        # Filter out None graphs (if any)
        batch = [item for item in batch if item[0] is not None]
        if len(batch) == 0:
            return None, None

        graphs, labels = zip(*batch)

        def stack_and_convert(key, dtype=None):
            arrs = [g[key] for g in graphs]
            if dtype is not None:
                arrs = [a.astype(dtype) for a in arrs]
            return torch.tensor(np.stack(arrs))

        coords = stack_and_convert('coords', np.float32)
        charges = stack_and_convert('charges', np.float32)
        sadic = stack_and_convert('sadic', np.float32)
        sasa = stack_and_convert('sasa', np.float32)
        mask = stack_and_convert('mask', np.float32)
        ca = stack_and_convert('ca', np.float32)

        # Convert atom_types using fixed element mapping
        atom_types_list = [g['atom_types'] for g in graphs]
        atom_types_idx = []
        for arr in atom_types_list:
            arr_idx = [_element_mapping.get(str(x), 8) for x in arr]  # 8 is unknown element
            atom_types_idx.append(arr_idx)
        atom_types = torch.tensor(np.stack(atom_types_idx), dtype=torch.long)

        labels = torch.tensor(labels, dtype=torch.long)

        batch_graph = {
            'coords': coords,
            'charges': charges,
            'sadic': sadic,
            'sasa': sasa,
            'mask': mask,
            'atom_types': atom_types,
            'ca': ca,
        }

        return batch_graph, labels

if __name__ == "__main__":
    dataset = GraphTransformerDataset('/mnt/beegfs/home/giulio/transformerSADIC/frank/data/masked_prediction/training_structural_info_with_SADIC.hdf5', max_entries=10)
    print(f"Dataset length: {len(dataset)}")
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
    for batch in dataloader:
        graphs, labels = batch
        print(f"Batch graphs: {graphs}")
        print(f"Batch labels: {labels}")
        break