# from MolLoaderSelfies import *
# import selfies as sf
# smiles = "CN1C(=O)C2=C(c3cc4c(s3)-c3sc(-c5ncc(C#N)s5)cc3C43OCCO3)N(C)C(=O)" \
#          "C2=C1c1cc2c(s1)-c1sc(-c3ncc(C#N)s3)cc1C21OCCO1"
# encoded_selfies = sf.encoder(smiles)  # SMILES --> SEFLIES
# decoded_smiles = sf.decoder(encoded_selfies)  # SELFIES --> SMILES
# default_constraints = sf.get_semantic_constraints()

# print(default_constraints)
# print(f"Original SMILES: {smiles}")
# print(f"Translated SELFIES: {encoded_selfies}")
# print(f"Translated SMILES: {decoded_smiles}")
# dataset = SMILESDataset("data/smiles_10000_selected_features.csv")

# from ProtLigDataset import *
# dataset = ProtLigDataset("data/protein_embeddings.npy", "data/smiles_output_selfies_normal.npy")
# print(dataset[0][1].shape)


# from rdkit import Chem
# from rdkit.Chem import SaltRemover

# remover = SaltRemover.SaltRemover()

# def remove_salts(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is not None:
#         mol = remover.StripMol(mol)
#         return Chem.MolToSmiles(mol)
#     return None

# print(remove_salts("CC[N+](CC)(CC)Cc1c2ccoc2c(OC)c2oc(=O)ccc12.[Cl-]"))


# from MolLoaderSelfies import SMILESDataset
# dataset = SMILESDataset("data/smiles_10000_selected_features_cleaned.csv", vocab_size=1000, max_length=128, tokenizer_path=None)
# print(len(dataset.tokenizer.token_to_id))

from pIC50Predictor2 import pIC50Predictor
import torch
device="cuda"
pic50_model = pIC50Predictor(1000, 1280).to(device)
pic50_model.load_state_dict(torch.load('models/pIC50_model.pt', map_location=device))
x = torch.randn(1, 256, 128, device=device)
prot = torch.randn(1, 1280, device=device)
t = torch.randint(0, 1000, (1,), device=device)
# x.requires_grad = True
print(x.requires_grad)
print(x)
print(prot)
print(t)
output = pic50_model(x, prot, t)
print(output.requires_grad)
print(pic50_model.training)