import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover

# Define the list of allowed elements
allowed_elements = set(['H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Se', 'Br', 'I'])

# Custom salt remover to ensure specific salts are removed
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

def process_smiles(df):
    processed_smiles = []
    indices_to_keep = []

    for i, smiles in enumerate(df['SMILES String']):
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
        indices_to_keep.append(i)
    
    # Update the DataFrame to keep only the valid rows
    df = df.iloc[indices_to_keep].copy()
    df['SMILES String'] = processed_smiles
    return df

# Example usage with the provided columns
df = pd.read_csv("data/protein_drug_pairs_with_sequences_and_smiles2.csv")

# Process the SMILES strings and update the DataFrame
df = process_smiles(df)

# Save the cleaned DataFrame
df.to_csv("data/protein_drug_pairs_with_sequences_and_smiles_cleaned2.csv", index=False)
