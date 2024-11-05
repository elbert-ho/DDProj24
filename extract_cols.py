import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Load the BindingDB TSV file into a DataFrame
file_path = '/Users/Elbert/Downloads/BindingDB_All.tsv'  # Replace with the actual path to your downloaded TSV file
df = pd.read_csv(file_path, sep='\t')

# Inspect the column names to ensure you have the correct columns
print(df.columns)

# Replace with your actual column names for SMILES, Protein Sequence, Kd, and Ki
ligand_smiles_column = 'Ligand SMILES'  # SMILES strings for the ligands
protein_sequence_column = 'BindingDB Target Chain Sequence'  # Protein sequences
kd_column = 'Kd (nM)'  # Kd values in nanomolar (nM)
ki_column = 'Ki (nM)'  # Ki values in nanomolar (nM)

# Step 1: Replace missing Kd values with Ki if available

def is_numeric(value):
    try:
        float(value)  # Try to convert to a float
        return True
    except ValueError:
        return False

df = df[df[ki_column].apply(lambda x: is_numeric(str(x)))]

# Convert Ki and Kd columns to numeric
df[ki_column] = pd.to_numeric(df[ki_column], errors='coerce')
df[kd_column] = pd.to_numeric(df[kd_column], errors='coerce')

# If Kd is missing but Ki is present, use Ki as an approximation for Kd
df[kd_column] = df[kd_column].fillna(df[ki_column])

# Step 2: Drop rows where both Kd and Ki are missing
df = df.dropna(subset=[kd_column])

# Step 3: Convert SMILES to Morgan fingerprints using RDKit
# def smiles_to_morgan(smiles, radius=2, n_bits=1024):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#     fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
#     return fingerprint

# Apply the function to convert SMILES to Morgan fingerprints
# df['Morgan Fingerprint'] = df[ligand_smiles_column].apply(lambda x: smiles_to_morgan(x))

# Drop rows where the SMILES conversion failed
# df = df.dropna(subset=['Morgan Fingerprint'])

# Step 4: Create the final dataset with the relevant columns
# Columns: Morgan Fingerprint, Protein Sequence, and Kd (which could be Ki if Kd was missing)
final_df = df[[ligand_smiles_column, protein_sequence_column, kd_column]]

# Optional: Convert final_df to a simpler format for storage (e.g., saving it to a CSV or JSON file)
final_df.to_csv('final_binding_affinities_dataset.csv', index=False)

# Preview the final dataset
print(final_df.head())
