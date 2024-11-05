from rdkit import Chem
from rdkit.Chem import AllChem, QED
import os
import sys
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'share', 'RDKit', 'Contrib', 'SA_Score'))
import sascorer

smile = "CC1([C@@H]2[C@H]1[C@H](N(C2)C(=O)[C@H](C(C)(C)C)NC(=O)C(F)(F)F)C(=O)N[C@@H](C[C@@H]3CCNC3=O)C#N)C"
mol = Chem.MolFromSmiles(smile)

qed = QED.qed(mol)
sas = sascorer.calculateScore(mol)

print(f"QED: {qed} SAS: {sas}")

# Generate 3D coordinates (optional, if you want to have a 3D structure in the SDF)
AllChem.EmbedMolecule(mol, AllChem.ETKDG())
# Write the molecule to an SDF file
with Chem.SDWriter(f'ligand.sdf') as writer:
    writer.write(mol)