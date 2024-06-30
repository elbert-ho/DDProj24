from chembl_webresource_client.new_client import new_client
import csv

def fetch_smiles_pagination(limit, output_file):
    molecule = new_client.molecule
    smiles_data = []
    offset = 0

    while len(smiles_data) < limit:
        molecules = molecule.filter(offset=offset, limit=100)
        
        for mol in molecules:
            if 'molecule_structures' in mol and mol['molecule_structures']:
                smiles = mol['molecule_structures'].get('canonical_smiles')
                if smiles:
                    smiles_data.append((mol['molecule_chembl_id'], smiles))
            if len(smiles_data) >= limit:
                break
        
        offset += 100
        print(f"Fetched {offset} compounds")

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ChEMBL_ID", "SMILES"])
        writer.writerows(smiles_data)

    print(f"Saved SMILES to {output_file}")

# Fetch 10,000 SMILES
fetch_smiles_pagination(500000, '../data/smiles_data1000.csv')
