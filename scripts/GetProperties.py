import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.Lipinski import NumHDonors, NumHAcceptors

# ESOL (Estimated Solubility) model
def esol(mol):
    if mol is None:
        return None
    logP = MolLogP(mol)
    mol_weight = Descriptors.MolWt(mol)
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    aromatic_atoms = sum([atom.GetIsAromatic() for atom in mol.GetAtoms()])
    # heavy_atoms = mol.GetNumHeavyAtoms()

    # ESOL equation based on Delaney (2004)
    solubility = (0.16 - 0.63 * logP - 0.0062 * mol_weight +
                  0.066 * num_rotatable_bonds - 0.74 * aromatic_atoms)
    return solubility

def calculate_properties(smiles):
    # try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None, None, None
        logP = MolLogP(mol)
        tpsa = CalcTPSA(mol)
        h_donors = NumHDonors(mol)
        h_acceptors = NumHAcceptors(mol)
        solubility = esol(mol)
        return logP, tpsa, h_donors, h_acceptors, solubility
    # except:
    #     return None, None, None, None, None

def add_properties_to_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    properties = df['SMILES'].apply(calculate_properties)
    df['logP'], df['tpsa'], df['h_donors'], df['h_acceptors'], df['solubility'] = zip(*properties)
    df.dropna(subset=['logP', 'tpsa', 'h_donors', 'h_acceptors', 'solubility'], inplace=True)
    df.to_csv(output_csv, index=False)

input_csv = 'chembl_smiles.csv'
output_csv = 'smiles_10000_with_props.csv'
add_properties_to_csv(input_csv, output_csv)
