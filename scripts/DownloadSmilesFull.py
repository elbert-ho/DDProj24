import requests
import pandas as pd
from tqdm import tqdm
import logging
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.Lipinski import NumHDonors, NumHAcceptors

def fetch_chembl_smiles(batch_size=100, max_records=50000):
    base_url = 'https://www.ebi.ac.uk/chembl/api/data/molecule'
    params = {
        'format': 'json',
        'limit': batch_size,
        'offset': 0,
        'molecule_properties__isnull': False,
        'molecule_structures__isnull': False
    }

    all_smiles = []
    with tqdm(total=max_records, desc="Fetching SMILES") as pbar:
        while len(all_smiles) < max_records:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            for molecule in data['molecules']:
                if 'molecule_structures' in molecule and 'canonical_smiles' in molecule['molecule_structures']:
                    all_smiles.append(molecule['molecule_structures']['canonical_smiles'])
                    pbar.update(1)
                    pbar.refresh()
                    if len(all_smiles) >= max_records:
                        break

            params['offset'] += batch_size

            if not data['page_meta']['next']:
                break

    return all_smiles[:max_records]

def save_smiles_to_csv(smiles_list, filename='../data/smiles_50000.csv'):
    df = pd.DataFrame(smiles_list, columns=['SMILES'])
    df.to_csv(filename, index=False)
    logging.info(f"Saved {len(smiles_list)} SMILES strings to {filename}")

# Fetching SMILES from ChEMBL
smiles = fetch_chembl_smiles()

# Saving to CSV file
save_smiles_to_csv(smiles)
print(f"{len(smiles)} SMILES strings have been saved")



# ESOL (Estimated Solubility) model
def esol(mol):
    if mol is None:
        return None
    logP = MolLogP(mol)
    mol_weight = Descriptors.MolWt(mol)
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    aromatic_atoms = sum([atom.GetIsAromatic() for atom in mol.GetAtoms()])
    heavy_atoms = mol.GetNumHeavyAtoms()

    # ESOL equation based on Delaney (2004)
    solubility = (0.16 - 0.63 * logP - 0.0062 * mol_weight +
                  0.066 * num_rotatable_bonds - 0.74 * aromatic_atoms) / heavy_atoms
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

input_csv = '../data/smiles_50000.csv'
output_csv = '../data/smiles_50000_with_props.csv'
add_properties_to_csv(input_csv, output_csv)
