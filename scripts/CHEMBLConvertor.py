import pandas as pd
from chembl_webresource_client.new_client import new_client
import requests
from tqdm import tqdm

# Initialize the ChEMBL client
molecule = new_client.molecule
target = new_client.target

# Function to get protein sequence from UniProt
def get_uniprot_sequence(accession):
    url = f"https://www.uniprot.org/uniprot/{accession}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_data = response.text
        sequence = ''.join(fasta_data.split('\n')[1:])  # Skip the header line
        return sequence
    else:
        print(f"Error fetching sequence for accession {accession} from UniProt")
        return None

# Function to get protein sequence and SMILES string
def get_protein_and_smiles(protein_chembl_id, drug_chembl_id):
    try:
        # Get protein details
        target_data = target.get(protein_chembl_id)
        if 'target_components' in target_data and target_data['target_components']:
            accession = target_data['target_components'][0]['accession']
            protein_sequence = get_uniprot_sequence(accession)
        else:
            print(f"Missing target components for {protein_chembl_id}")
            return None, None

        # Get drug details
        molecule_data = molecule.get(drug_chembl_id)
        if 'molecule_structures' in molecule_data and molecule_data['molecule_structures']:
            smiles_string = molecule_data['molecule_structures']['canonical_smiles']
        else:
            print(f"Missing molecule structures for {drug_chembl_id}")
            return None, None

        return protein_sequence, smiles_string
    except Exception as e:
        print(f"Error fetching data for {protein_chembl_id} or {drug_chembl_id}: {e}")
        return None, None

# Read the input CSV
input_df = pd.read_csv('../data/protein_drug_pairs2.csv')

# Prepare lists for the new columns
protein_sequences = []
smiles_strings = []

# Fetch data for each row
for index, row in tqdm(input_df.iterrows(), total=len(input_df)):
    protein_chembl_id = row['Protein ChEMBL ID']
    drug_chembl_id = row['Drug ChEMBL ID']
    
    protein_sequence, smiles_string = get_protein_and_smiles(protein_chembl_id, drug_chembl_id)
    
    if protein_sequence and smiles_string:
        protein_sequences.append(protein_sequence)
        smiles_strings.append(smiles_string)
    else:
        protein_sequences.append(None)
        smiles_strings.append(None)

# Create the output DataFrame
output_df = pd.DataFrame({
    'Protein Sequence': protein_sequences,
    'SMILES String': smiles_strings
})

# Save to CSV
output_df.to_csv('../data/protein_drug_pairs_with_sequences_and_smiles2.csv', index=False)
print("Converted data has been saved to 'protein_drug_pairs_with_sequences_and_smiles.csv'.")
