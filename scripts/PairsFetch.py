import pandas as pd
from chembl_webresource_client.new_client import new_client
import numpy as np

# Initialize the ChEMBL target and activity clients
target = new_client.target
activity = new_client.activity

# Define a function to get protein-drug pairs with IC50 values and convert to pIC50
def get_protein_drug_pairs(limit=100, max_pairs_per_protein=5):
    # Search for human proteins
    targets = target.filter(organism__icontains='Homo sapiens')
    
    protein_drug_pairs = []
    protein_pair_count = {}
    target_count = 0
    
    for t in targets:
        if len(protein_drug_pairs) >= limit:
            break
        
        target_chembl_id = t['target_chembl_id']
        target_count += 1
        print(f"Processing target {target_count}: {target_chembl_id}")
        
        if protein_pair_count.get(target_chembl_id, 0) >= max_pairs_per_protein:
            continue
        
        try:
            # Get activities for the target
            activities = activity.filter(target_chembl_id=target_chembl_id, standard_type='IC50', limit=100)  # Limit to avoid timeouts
            
            for act in activities:
                if len(protein_drug_pairs) >= limit:
                    break
                if protein_pair_count.get(target_chembl_id, 0) >= max_pairs_per_protein:
                    break
                if 'standard_value' in act and act['standard_value'] is not None:
                    IC50_value = act['standard_value']
                    molecule_chembl_id = act['molecule_chembl_id']
                    
                    # Convert IC50 to pIC50
                    try:
                        pIC50_value = -np.log10(float(IC50_value) * 1e-9)
                    except ValueError:
                        print(f"Invalid IC50 value: {IC50_value} for molecule {molecule_chembl_id}")
                        continue
                    
                    protein_drug_pairs.append({
                        'Protein ChEMBL ID': target_chembl_id,
                        'Drug ChEMBL ID': molecule_chembl_id,
                        'pIC50': pIC50_value
                    })
                    
                    # Update the count for this protein
                    protein_pair_count[target_chembl_id] = protein_pair_count.get(target_chembl_id, 0) + 1
        except Exception as e:
            print(f"Error processing target {target_chembl_id}: {e}")
                
    return protein_drug_pairs

# Specify the limit for the number of pairs and max pairs per protein
limit = 10000  # Change this value to control the number of pairs
max_pairs_per_protein = 50  # Change this value to control the number of pairs per protein

# Get the protein-drug pairs
protein_drug_pairs = get_protein_drug_pairs(limit=limit, max_pairs_per_protein=max_pairs_per_protein)

# Convert list of dictionaries to DataFrame
protein_drug_df = pd.DataFrame.from_records(protein_drug_pairs)
protein_drug_df.to_csv('../protein_drug_pairs.csv', index=False)
print(f"Protein-drug pairs with pIC50 values have been saved to 'protein_drug_pairs.csv'. Number of pairs: {len(protein_drug_df)}")
