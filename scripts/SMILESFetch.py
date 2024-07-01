from chembl_webresource_client.new_client import new_client
import csv
from tqdm import tqdm

def fetch_smiles_pagination(limit, output_file):
    molecule = new_client.molecule
    smiles_data = []
    offset = 0
    batch_size = 100

    with tqdm(total=limit) as pbar:
        while len(smiles_data) < limit:
            molecules = molecule.filter(offset=offset, limit=batch_size)
            for mol in molecules:
                # print(mol)
                if 'molecule_structures' in mol and mol['molecule_structures']:
                    smiles = mol['molecule_structures'].get('canonical_smiles')
                    if smiles and (mol['molecule_chembl_id'], smiles) not in smiles_data:
                        smiles_data.append((mol['molecule_chembl_id'], smiles))
                        pbar.update(1)  # Update the progress bar for each unique molecule added
                        pbar.refresh()  # Refresh the progress bar
                if len(smiles_data) >= limit:
                    break
            
            offset += batch_size

            # Break the loop if no more molecules are returned
            if not molecules:
                break

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ChEMBL_ID", "SMILES"])
        writer.writerows(smiles_data)

    print(f"Saved SMILES to {output_file}")

# Fetch 10,000 SMILES
fetch_smiles_pagination(10000, '../data/smiles_data10000.csv')
