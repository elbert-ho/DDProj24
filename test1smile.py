from rdkit import Chem
from rdkit.Chem import AllChem
import os
import sys
import os
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'share', 'RDKit', 'Contrib', 'SA_Score'))
import sascorer
from rdkit.Chem import Descriptors, QED

smile = "CC1(C2C1C(N(C2)C(=O)C(C(C)(C)C)NC(=O)C(F)(F)F)C(=O)NC(CC3CCNC3=O)C#N)C"
if smile is not None and smile != "":
        mol = Chem.MolFromSmiles(smile)
        print(QED.qed(mol))
        print(sascorer.calculateScore(mol))
        # Generate 3D coordinates (optional, if you want to have a 3D structure in the SDF)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        # Write the molecule to an SDF file
        with Chem.SDWriter(f'ligand_test.sdf') as writer:
            writer.write(mol)