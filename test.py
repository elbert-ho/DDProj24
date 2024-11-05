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

# from TestingUtils import *
# import pandas as pd

# def apply_similarity_calculation(group):
#     smiles_list = group['SMILES String'].tolist()
#     return np.mean(calculate_internal_pairwise_similarities(smiles_list))

# df = pd.read_csv("data/protein_drug_pairs_with_sequences_and_smiles_cleaned.csv")
# results = df.groupby('Protein Sequence').apply(apply_similarity_calculation)

# # print(results)

# # Find the protein sequence with the maximum similarity value
# max_similarity = results.idxmax()  # Protein sequence with the max similarity
# max_value = results.max()  # Maximum similarity value

# # Display the results
# print(f"The protein sequence with the maximum similarity is: {max_similarity}")
# print(f"The maximum similarity value is: {max_value}")

# # Sort the results by similarity values
# sorted_results = results.sort_values()

# # Get the index of the median value
# median_index = len(sorted_results) // 2
# median_protein_sequence = sorted_results.index[median_index]
# median_similarity_value = sorted_results.iloc[median_index]
# # Display the results
# print(f"The protein sequence with the median similarity is: {median_protein_sequence}")
# print(f"The median similarity value is: {median_similarity_value}")


# # Find the protein sequence with the minimum similarity value
# min_similarity_protein = results.idxmin()  # Protein sequence with the minimum similarity
# min_similarity_value = results.min()  # Minimum similarity value

# # Display the results
# print(f"The protein sequence with the minimum similarity is: {min_similarity_protein}")
# print(f"The minimum similarity value is: {min_similarity_value}")

# sims = calculate_internal_pairwise_similarities(df["SMILES String"][9653:9703]).flatten()
# import matplotlib.pyplot as plt

# # Plot histogram of the median similarity values
# # plt.figure(figsize=(8, 6))
# # plt.hist(sims, bins=20, color='skyblue', edgecolor='black')
# # plt.xlabel('Median Similarity')
# # plt.ylabel('Frequency')
# # plt.title('Histogram of Pairwise Similarity Values')
# # plt.tight_layout()
# # plt.show()

# # import matplotlib.pyplot as plt

# # Plot histogram of the median similarity values
# plt.figure(figsize=(8, 6))
# plt.hist(results.values, bins=10, color='skyblue', edgecolor='black')
# plt.xlabel('Median Similarity')
# plt.ylabel('Frequency')
# plt.title('Histogram of Pairwise Similarity Values')
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import numpy as np
# df = pd.read_csv("data/protein_drug_pairs_with_sequences_and_smiles_cleaned.csv")
# # arr = df["SMILES String"].to_numpy().tobytes()
# # # Search for strings containing "."
# # contains_dot = np.char.find(arr, '.') >= 0

# # # Get the strings that contain "."
# # result = arr[contains_dot]

# # print(result)

# # Example DataFrame column with SMILES strings
# arr = df["SMILES String"].to_numpy().astype(str)

# # print(arr)

# # Search for strings containing "."
# contains_dot = np.char.find(arr, '.') >= 0

# # print(contains_dot)

# # # Get the strings that contain "."
# result = arr[contains_dot]

# print(result)

# import pandas as pd
# import numpy as np

# # Load your CSV file into a DataFrame
# df = pd.read_csv("data/smiles_10000_selected_features_cleaned.csv")

# # Convert the "SMILES String" column to a string data type
# df["SMILES"] = df["SMILES"].astype(str)

# # Search for strings in the "SMILES String" column that contain "."
# contains_dot = df["SMILES"].str.contains('\.')

# # Remove rows where the "SMILES String" contains "."
# df_cleaned = df[~contains_dot]



# # Save the cleaned DataFrame to a new CSV file
# df_cleaned.to_csv("data/smiles_10000_no_dots.csv", index=False)

# print(f"Rows with '.' removed. New DataFrame saved to 'data/smiles_10000_no_dots.csv'.")

# import pandas as pd
# import numpy as np

# # Load your first CSV file into a DataFrame
# df1 = pd.read_csv("data/smiles_10000_selected_features_cleaned.csv")

# # Load your second CSV file into a DataFrame
# df2 = pd.read_csv("data/protein_drug_pairs_with_sequences_and_smiles_cleaned.csv")

# # Convert the "SMILES" column in both DataFrames to string type
# df1["SMILES"] = df1["SMILES"].astype(str)
# df2["SMILES"] = df2["SMILES String"].astype(str)

# # Remove rows where "SMILES" contains a "." in both DataFrames
# df1_cleaned = df1[~df1["SMILES"].str.contains('\.')]
# df2_cleaned = df2[~df2["SMILES"].str.contains('\.')]

# # Combine/merge the cleaned SMILES columns from both DataFrames into one
# merged_df = pd.concat([df1_cleaned["SMILES"], df2_cleaned["SMILES String"]], ignore_index=True)

# # Save the merged DataFrame to a new CSV file
# merged_df.to_csv("data/merged_smiles_no_dots.csv", index=False, header=["SMILES"])

# print(f"Merged DataFrame with cleaned SMILES saved to 'data/merged_smiles_no_dots.csv'.")

# import torch
# from MolLoaderSelfiesFinal import SMILESDataset
# from tqdm import tqdm 


# import selfies as sf

# # Load the dataset
# file_path = "data/smiles_10000_selected_features_cleaned.csv"
# # file_path = "data/merged_smiles_no_dots.csv"
# dataset = SMILESDataset(file_path, vocab_size=512, max_length=256)

# # Iterate over the dataset and find the maximum encoding length
# max_length = 0

# count = 0
# for i in tqdm(range(len(dataset))):  # Wrap the range with tqdm
#     ids, _ = dataset[i]  # Get the tokenized sequence
#     length = len(ids.nonzero(as_tuple=True)[0])  # Count the non-padding tokens
#     if length > max_length:
#         max_length = length

#     if length > 128:
#         count += 1
    
# print(count)

# print(f"The maximum encoding length in the dataset is: {max_length}")


# ________

# import csv
# import torch
# from MolLoaderSelfiesFinal import SMILESDataset

# # Load the dataset
# # file_path = "data/protein_drug_pairs_with_sequences_and_smiles_cleaned.csv"

# file_path = "data/protein_drug_pairs_with_sequences_and_smiles_cleaned2.csv"

# output_file_path = "data/pd_truncated_final_2.csv"  # New file to save filtered data
# # dataset0 = SMILESDataset("data/smiles_10000_selected_features_cleaned.csv", vocab_size=v_size, max_length=128)

# dataset = SMILESDataset(file_path, vocab_size=256, max_length=256, props=False, tokenizer_path="models/selfies_tokenizer_final.json", unicode_path="models/unicode_mapping.json")
# # dataset = SMILESDataset(file_path, vocab_size=256, max_length=256, props=False)


# # Open the original CSV file to read and filter rows
# with open(file_path, 'r') as input_file:
#     reader = csv.DictReader(input_file)
#     rows = list(reader)

# # List to hold valid rows
# valid_rows = []
# # Iterate over the dataset and check if the tokenized length is <= 128
# for i in range(len(dataset)):
#     ids, _ = dataset[i]  # Get the tokenized sequence
#     if ids is None:
#         continue
#     length = len(ids.nonzero(as_tuple=True)[0])  # Count the non-padding tokens
    
#     if length <= 128:
#         valid_rows.append(rows[i])  # Keep the row if the sequence length is valid

# # Write the valid rows to the new CSV file
# with open(output_file_path, 'w', newline='') as output_file:
#     writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
#     writer.writeheader()
#     writer.writerows(valid_rows)

# print(f"Filtered dataset saved to {output_file_path}. Number of valid sequences: {len(valid_rows)}")


# Regularize the SMILES
