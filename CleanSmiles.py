import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover

# Define the list of allowed elements
allowed_elements = set(['H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Se', 'Br', 'I'])

# Custom salt remover to ensure specific salts are removed
# Use the default SaltRemover
remover = SaltRemover.SaltRemover()

def remove_salts(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mol = remover.StripMol(mol)
        return Chem.MolToSmiles(mol)
    return None

def neutralize_charges(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        pattern = Chem.MolFromSmarts("[+1!H0,-1!H0]")
        at_matches = mol.GetSubstructMatches(pattern)
        for match in at_matches:
            atom = mol.GetAtomWithIdx(match[0])
            if atom.GetFormalCharge() > 0:
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(atom.GetTotalNumHs() - 1)
            elif atom.GetFormalCharge() < 0:
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
        return Chem.MolToSmiles(mol)
    return None

def contains_only_allowed_elements(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in allowed_elements:
                return False
        return True
    return False

def process_smiles(smiles_list):
    processed_smiles = []
    for smiles in smiles_list:
        # Remove salts
        smiles = remove_salts(smiles)
        if smiles is None:
            continue
        
        # Neutralize charges
        smiles = neutralize_charges(smiles)
        if smiles is None:
            continue
        
        # Check for allowed elements
        if not contains_only_allowed_elements(smiles):
            continue
        
        processed_smiles.append(smiles)
    return processed_smiles

# Example usage with the provided columns
# df = pd.read_csv("data/smiles_10000_selected_features.csv")
df = pd.read_csv("data/protein_drug_pairs_with_sequences_and_smiles.csv")


# Process the SMILES strings and add to a new column
df['SMILES String'] = process_smiles(df['SMILES String'])

# Filter out rows with invalid SMILES
df = df[df['SMILES String'].notna()]
# Example usage
df.to_csv("data/protein_drug_pairs_with_sequences_and_smiles_cleaned.csv", index=False)