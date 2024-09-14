import pandas as pd
from rdkit import Chem

# Function to check if SMILES is valid
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

# Load the dataset
input_csv = 'data/final_binding_affinities_dataset.csv'
output_csv = 'data/cleaned_binding_affinities_dataset.csv'

# Load the CSV into a pandas DataFrame
df = pd.read_csv(input_csv)

# Apply the SMILES validity check
df['valid_smiles'] = df['SMILES'].apply(is_valid_smiles)

# Filter out rows with invalid SMILES
cleaned_df = df[df['valid_smiles']].drop(columns=['valid_smiles'])

# Save the cleaned DataFrame to a new CSV file
cleaned_df.to_csv(output_csv, index=False)

print(f"Preprocessing complete. Valid SMILES saved to {output_csv}")
