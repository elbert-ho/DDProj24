from TestingUtils import calculate_pairwise_similarities, calculate_internal_pairwise_similarities
import pandas as pd
import numpy as np

def apply_similarity(group):
    smiles_strings = group['SMILES String'].tolist()
    similarity = calculate_internal_pairwise_similarities(smiles_strings)
    return np.mean(similarity)

df = pd.read_csv("data/protein_drug_pairs_with_sequences_and_smiles_cleaned.csv")
smiles = df["SMILES String"][150:200]

print(np.mean(calculate_internal_pairwise_similarities(smiles)))

# Group by 'Protein' and apply the similarity function
grouped = df.groupby('Protein Sequence').apply(apply_similarity)

# print(grouped)

# Calculate the overall average similarity across all proteins
overall_average_similarity = grouped.mean()
print(overall_average_similarity)

# smiles = df["SMILES String"]
# print(np.mean(calculate_internal_pairwise_similarities(smiles)))
