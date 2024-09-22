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

# from pIC50Predictor2 import pIC50Predictor
# import torch
# device="cuda"
# pic50_model = pIC50Predictor(1000, 1280).to(device)
# pic50_model.load_state_dict(torch.load('models/pIC50_model.pt', map_location=device))
# x = torch.randn(1, 256, 128, device=device)
# prot = torch.randn(1, 1280, device=device)
# t = torch.randint(0, 1000, (1,), device=device)
# # x.requires_grad = True
# print(x.requires_grad)
# print(x)
# print(prot)
# print(t)
# output = pic50_model(x, prot, t)
# print(output.requires_grad)
# print(pic50_model.training)

from TestingUtils import *
import pandas as pd

def apply_similarity_calculation(group):
    smiles_list = group['SMILES String'].tolist()
    return np.mean(calculate_internal_pairwise_similarities(smiles_list))

df = pd.read_csv("data/protein_drug_pairs_with_sequences_and_smiles_cleaned.csv")
results = df.groupby('Protein Sequence').apply(apply_similarity_calculation)

# print(results)

# Find the protein sequence with the maximum similarity value
max_similarity = results.idxmax()  # Protein sequence with the max similarity
max_value = results.max()  # Maximum similarity value

# Display the results
print(f"The protein sequence with the maximum similarity is: {max_similarity}")
print(f"The maximum similarity value is: {max_value}")

# Sort the results by similarity values
sorted_results = results.sort_values()

# Get the index of the median value
median_index = len(sorted_results) // 2
median_protein_sequence = sorted_results.index[median_index]
median_similarity_value = sorted_results.iloc[median_index]
# Display the results
print(f"The protein sequence with the median similarity is: {median_protein_sequence}")
print(f"The median similarity value is: {median_similarity_value}")


# Find the protein sequence with the minimum similarity value
min_similarity_protein = results.idxmin()  # Protein sequence with the minimum similarity
min_similarity_value = results.min()  # Minimum similarity value

# Display the results
print(f"The protein sequence with the minimum similarity is: {min_similarity_protein}")
print(f"The minimum similarity value is: {min_similarity_value}")

sims = calculate_internal_pairwise_similarities(df["SMILES String"][9653:9703]).flatten()
import matplotlib.pyplot as plt

# Plot histogram of the median similarity values
# plt.figure(figsize=(8, 6))
# plt.hist(sims, bins=20, color='skyblue', edgecolor='black')
# plt.xlabel('Median Similarity')
# plt.ylabel('Frequency')
# plt.title('Histogram of Pairwise Similarity Values')
# plt.tight_layout()
# plt.show()

# import matplotlib.pyplot as plt

# Plot histogram of the median similarity values
plt.figure(figsize=(8, 6))
plt.hist(results.values, bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Median Similarity')
plt.ylabel('Frequency')
plt.title('Histogram of Pairwise Similarity Values')
plt.tight_layout()
plt.show()