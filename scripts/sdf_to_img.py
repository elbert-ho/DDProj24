from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

# Load the molecule from the SDF file
suppl = Chem.SDMolSupplier('../ligands/ligand097.sdf')
mol = suppl[0]

if mol is None:
    raise ValueError("Could not read the molecule from the SDF file.")

# Convert the molecule to a SMILES string
smiles = Chem.MolToSmiles(mol)
print(f'SMILES: {smiles}')

# Convert SMILES back to a molecule object to ensure 2D representation
mol_2d = Chem.MolFromSmiles(smiles)

# Generate 2D coordinates
AllChem.Compute2DCoords(mol_2d)

# Draw the 2D image and save it
img = Draw.MolToImage(mol_2d)
img.save('../ligand097.png')
