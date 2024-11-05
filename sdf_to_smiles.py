from rdkit import Chem

# Load the SDF file
sdf_file = "ligand_temp.sdf"  # Replace with your SDF file path
suppl = Chem.SDMolSupplier(sdf_file)

# Convert each molecule to SMILES and print
smiles_list = []
for mol in suppl:
    if mol is not None:
        smiles = Chem.MolToSmiles(mol)
        smiles_list.append(smiles)
        print(smiles)
