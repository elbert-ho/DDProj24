from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
import pandas as pd

smiles = pd.read_csv("../data/protein_drug_pairs_with_sequences_and_smiles_cleaned.csv")["SMILES String"].to_list()[100:150]
idx = 0
for smile in smiles:
    # Convert SMILES to a molecule object
    mol = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    # Generate 3D coordinates (optional, if you want to have a 3D structure in the SDF)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    # Write the molecule to an SDF file
    with Chem.SDWriter(f'../ligands_known/ligand{idx:03}.sdf') as writer:
        writer.write(mol)
    idx += 1
