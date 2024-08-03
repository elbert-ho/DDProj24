# IMPORTS
import torch
import yaml
from DiffusionModelGLIDE import *
from tqdm import tqdm
from unet_condition import Text2ImUNet
from transformers import EsmTokenizer, EsmModel
from MolTransformerSelfies import MultiTaskTransformer
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from fcd_torch import FCD
from SelfiesTok import SelfiesTok
import numpy as np
import pandas as pd
import random
import umap
import matplotlib.pyplot as plt
from typing import Set, Any
from TestingUtils import *
from scipy.stats import wasserstein_distance
import sys
import os
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'share', 'RDKit', 'Contrib', 'SA_Score'))
import sascorer
from rdkit.Chem import Descriptors, QED
from collections import Counter
import selfies as sf

# Step 0: Load and generate
# Step 0.1: Load in the training data (potential source of issue as this includes val but whatever)
ref_file_smiles = "data/protein_drug_pairs_with_sequences_and_smiles.csv"
ref_file_finger = "data/smiles_output_selfies.npy"
ref_data_finger = np.load(ref_file_finger)
ref_data_smiles = pd.read_csv(ref_file_smiles)

# Step 0.2: Load in GaussianDiffusion model and unet model and MolTransformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

n_diff_step = config["diffusion_model"]["num_diffusion_steps"]
protein_embedding_dim = config["protein_model"]["protein_embedding_dim"]
batch_size = config["diffusion_model"]["batch_size"]
src_vocab_size = config["mol_model"]["src_vocab_size"]
tgt_vocab_size = config["mol_model"]["tgt_vocab_size"]
max_seq_length = config["mol_model"]["max_seq_length"]
num_tasks = config["mol_model"]["num_tasks"]
d_model = config["mol_model"]["d_model"]
num_heads = config["mol_model"]["num_heads"]
num_layers = config["mol_model"]["num_layers"]
d_ff = config["mol_model"]["d_ff"]
dropout = config["mol_model"]["dropout"]

diffusion_model = GaussianDiffusion(betas=get_named_beta_schedule(n_diff_step))
unet = Text2ImUNet(text_ctx=1, xf_width=protein_embedding_dim, xf_layers=0, xf_heads=0, xf_final_ln=0, tokenizer=None, in_channels=1, model_channels=48, out_channels=2, num_res_blocks=2, attention_resolutions=[], dropout=.1, channel_mult=(1, 2, 4, 8), dims=1)
mol_model = MultiTaskTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks).to(device)

unet.load_state_dict(torch.load('unet.pt', map_location=device))
mol_model.load_state_dict(torch.load('models/selfies_transformer.pt', map_location=device))

unet, mol_model = unet.to(device), mol_model.to(device)

# Step 0.3: Pick 40 random proteins and generate 25 molecules per protein
# Group by Protein Sequence and filter groups that have at least 25 SMILES strings
ref_data_smiles.reset_index(inplace=True)
protein_groups = ref_data_smiles.groupby('Protein Sequence').filter(lambda x: len(x) >= 25)

# Get unique protein sequences that have at least 25 SMILES strings
unique_proteins = protein_groups['Protein Sequence'].unique()

# Select 40 random unique proteins
random_proteins = random.sample(list(unique_proteins), 40)

# For each selected protein, extract 25 random SMILES strings
selected_rows = []
for protein in random_proteins:
    protein_df = protein_groups[protein_groups['Protein Sequence'] == protein]
    selected_smiles = protein_df.sample(25)
    selected_rows.append(selected_smiles)

ref_data_smiles_cut = pd.concat(selected_rows)

# Get ESM
protein_indices = ref_data_smiles_cut.loc[:,"index"].to_numpy()
protein_fingers_full = np.load("data/protein_embeddings.npy")
protein_fingers_cut = []
for protein_idx in protein_indices:
    protein_fingers_cut.append(protein_fingers_full[protein_idx])

# Step 0.3.1: Generate 25 molecules per protein
samples = torch.tensor([])
for protein_finger in protein_fingers_cut:
    protein_finger = torch.tensor(protein_finger, device=device)
    protein_finger = protein_finger.repeat(25, 1)
    sample = diffusion_model.p_sample_loop(unet, (25, 1, d_model * max_seq_length), prot=protein_finger).detach()
    samples = torch.cat([samples, sample])

# Debug line
print(samples.shape)

# Step 0.3.2: Remember to unnormalize
mins = torch.tensor(np.load("data/smiles_mins_selfies.npy")).reshape(1, 1, -1)
maxes = torch.tensor(np.load("data/smiles_maxes_selfies.npy")).reshape(1, 1, -1)
sample_rescale = (((sample + 1) / 2) * (maxes - mins) + mins)

# Step 0.3.3: Convert from SELFIES back to SMILES
tokenizer = SelfiesTok.load("models/selfies_tok.json")
with torch.no_grad():
    decoded_smiles, _ = mol_model.decode_representation(sample_rescale.reshape(-1, max_seq_length, d_model), None, max_length=128, tokenizer=tokenizer)

gen_smiles = []

for decode in decoded_smiles:
    predicted_selfie = tokenizer.decode(decode.detach().cpu().flatten().tolist(), skip_special_tokens=True)
    predicted_smile = sf.decoder(predicted_selfie)
    gen_smiles.append(predicted_smile)

# Step 0.4: Isolate those 20 proteins and their original 50 molecules as well
smiles_fingers_cut = []
smiles_fingers_full = np.load("smiles_output_selfies.npy")
for protein_idx in protein_indices:
    smiles_fingers_cut.append(smiles_fingers_full[protein_idx].tolist())

# Step 0.5: Plot in U-Map with 2 colors
gen_fingers = sample_rescale.cpu().tolist()
combined_data = gen_fingers + smiles_fingers_cut
# Create labels for coloring
labels = [0] * len(gen_fingers) + [1] * len(smiles_fingers_cut)

# Convert to numpy array
combined_data_np = np.array(combined_data)

# Fit UMAP
reducer = umap.UMAP()
embedding = reducer.fit_transform(combined_data_np)

# Plotting
plt.figure(figsize=(10, 7))
plt.scatter(embedding[labels == 0, 0], embedding[labels == 0, 1], c='blue', label='Generated')
plt.scatter(embedding[labels == 1, 0], embedding[labels == 1, 1], c='red', label='Original')
plt.legend()
plt.title('UMAP Projection')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.savefig("umap.png")

# Step 1: GuacMol
# Step 1.1: Compute validity, uniqueness, novelty, FCD, KL-divergence
# Step 1.1.1: Validity
gen_mols = []
valid = 0
total = 0
for gen_smile in gen_smiles:
    try:
        mol = Chem.MolFromSmiles(gen_smile)
    except:
        pass
    gen_mols.append(mol)

print(f"Validity: {valid} / {total}")

# Step 1.1.2: Uniqueness
# unique_mols: Set[Any] = set()
# unique_gen = []
# for element in gen_mols:
#     if element not in unique_mols:
#         unique_mols.add(element)
#         unique_gen.append(element)

unique_smiles_set = set(gen_smiles)
unique_smiles_list = list(unique_smiles_set)

print(f"Uniqueness: {len(unique_smiles_list)} / {total}")

# Step 1.1.3: Novelty
ref_mols = []
ref_smiles = []
ref_data_smiles_cut
for smiles in ref_data_smiles_cut.loc[:,"SMILES String"]:
    ref_smiles.append(smiles)
    mol = Chem.MolFromSmiles(smiles)
    ref_mols.append(mol)

novel_smiles = unique_smiles_set.difference(ref_smiles)
print(f"Novelty: {len(novel_smiles)} / {total}")

# Step 1.1.4: KL-divergence
pc_descriptor_subset = [
            'BertzCT',
            'MolLogP',
            'MolWt',
            'TPSA',
            'NumHAcceptors',
            'NumHDonors',
            'NumRotatableBonds',
            'NumAliphaticRings',
            'NumAromaticRings'
            ]

d_gen = calculate_pc_descriptors(novel_smiles, pc_descriptor_subset)
d_ref = calculate_pc_descriptors(ref_smiles, pc_descriptor_subset)
kldivs = {}

# now we calculate the kl divergence for the float valued descriptors ...
for i in range(4):
    kldiv = continuous_kldiv(X_baseline=d_ref[:, i], X_sampled=d_gen[:, i])
    kldivs[pc_descriptor_subset[i]] = kldiv

# ... and for the int valued ones.
for i in range(4, 9):
    kldiv = discrete_kldiv(X_baseline=d_ref[:, i], X_sampled=d_gen[:, i])
    kldivs[pc_descriptor_subset[i]] = kldiv

# pairwise similarity

chembl_sim = calculate_internal_pairwise_similarities(ref_smiles)
chembl_sim = chembl_sim.max(axis=1)

sampled_sim = calculate_internal_pairwise_similarities(unique_smiles_list)
sampled_sim = sampled_sim.max(axis=1)

kldiv_int_int = continuous_kldiv(X_baseline=chembl_sim, X_sampled=sampled_sim)
kldivs['internal_similarity'] = kldiv_int_int
partial_scores = [np.exp(-score) for score in kldivs.values()]
kl_score = sum(partial_scores) / len(partial_scores)
print(f"KL Divergence: {kl_score}")

# Step 1.1.5: FCD
fcd = FCD(device='cuda:0', n_jobs=8)
fcd_score = fcd(gen_smiles, ref_smiles)
print(f"FCD: {fcd_score}")

# Step 2: MOSES
# Step 2.1: Compute SNN, IntDiv_1, IntDiv_2
# Step 2.1.1: Calculate IntDiv
gen_sim_nonunique = calculate_internal_pairwise_similarities(gen_smiles)
int_div_1 = 1 - (1 / (total * total)) * np.sum(gen_sim_nonunique)
int_div_2 = 1 - np.sqrt((1 / (total * total)) * np.sum(np.square(gen_sim_nonunique)))
print(f"IntDiv1: {int_div_1}")
print(f"IntDiv2: {int_div_2}")

# Step 2.1.2: Compute SNN
pairwise_sim = calculate_pairwise_similarities(ref_smiles, gen_smiles)

snn = 1 / total * np.sum(np.max(pairwise_sim, axis=0))
print(f"SNN: {snn}")

# Step 2.2: Compute Wasserstein and plot MW, logP, SA, QED
# Step 2.2.1: MW
def compute_properties(mols):
    properties = {'MW': [], 'logP': [], 'SA': [], 'QED': []}
    for mol in mols:
        if mol is not None:
            properties['MW'].append(Descriptors.MolWt(mol))
            properties['logP'].append(Descriptors.MolLogP(mol))
            properties['SA'].append(sascorer.calculateScore(mol))
            properties['QED'].append(QED.qed(mol))
    return properties

# Compute properties for reference and generated molecules
ref_properties = compute_properties(ref_mols)
gen_properties = compute_properties(gen_mols)

# Compute Wasserstein distances
wasserstein_distances = {}
for prop in ref_properties:
    wasserstein_distances[prop] = wasserstein_distance(ref_properties[prop], gen_properties[prop])

print("Wasserstein Distances:")
for prop, dist in wasserstein_distances.items():
    print(f"{prop}: {dist}")

# Plot distributions
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

def plot_distribution(ax, ref_data, gen_data, title):
    ax.hist(ref_data, bins=30, alpha=0.5, label='Reference')
    ax.hist(gen_data, bins=30, alpha=0.5, label='Generated')
    ax.set_title(title)
    ax.legend()

plot_distribution(axs[0, 0], ref_properties['MW'], gen_properties['MW'], 'Molecular Weight')
plot_distribution(axs[0, 1], ref_properties['logP'], gen_properties['logP'], 'LogP')
plot_distribution(axs[1, 0], ref_properties['SA'], gen_properties['SA'], 'Surface Area')
plot_distribution(axs[1, 1], ref_properties['QED'], gen_properties['QED'], 'QED')

plt.tight_layout()
plt.savefig("descriptors.png")

# Step 2.3: Compute Frag, Scaff (implement later)

def compute_fragments(mols):
    fragments = []
    for mol in mols:
        if mol is None:
            continue
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        for frag in frags:
            fragments.append(Chem.MolToSmiles(frag))
    return fragments

def compute_scaffolds(mols):
    scaffolds = []
    for mol in mols:
        if mol is None:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.append(scaffold)
    return scaffolds

def compute_frag_score(real_frags, gen_frags):
    real_counter = Counter(real_frags)
    gen_counter = Counter(gen_frags)
    intersection = sum((real_counter & gen_counter).values())
    return intersection / len(gen_frags)

def compute_scaff_score(real_scaffs, gen_scaffs):
    real_counter = Counter(real_scaffs)
    gen_counter = Counter(gen_scaffs)
    intersection = sum((real_counter & gen_counter).values())
    return intersection / len(gen_scaffs)

def compute_moses_scores(ref_mols, gen_mols):
    real_frags = compute_fragments(ref_mols)
    gen_frags = compute_fragments(gen_mols)
    frag_score = compute_frag_score(real_frags, gen_frags)

    real_scaffs = compute_scaffolds(ref_mols)
    gen_scaffs = compute_scaffolds(gen_mols)
    scaff_score = compute_scaff_score(real_scaffs, gen_scaffs)

    return frag_score, scaff_score

frag_score, scaff_score = compute_moses_scores(ref_mols, gen_mols)
print(f"Frag Score: {frag_score}")
print(f"Scaff Score: {scaff_score}")

# Step 3: Experiments with real proteins
# Step 3.1: Load in real protein (3cl protease)
# cl3 = torch.tensor(np.load("data/3cl.npy"))

# # Step 3.2: Run model on this protein and generate 100 compounds

# sample_cl3 = diffusion_model.p_sample_loop(unet, (100, 1, d_model * max_seq_length), prot=cl3).detach()
# sample_cl3_rescale = (((sample + 1) / 2) * (maxes - mins) + mins)

# with torch.no_grad():
#     decoded_smiles_cl3, _ = mol_model.decode_representation(sample_cl3_rescale.reshape(-1, max_seq_length, d_model), None, max_length=128, tokenizer=tokenizer)

# gen_smiles_cl3 = []
# gen_mols_cl3 = []

# for decode in decoded_smiles_cl3:
#     predicted_selfie = tokenizer.decode(decode.detach().cpu().flatten().tolist(), skip_special_tokens=True)
#     predicted_smile = sf.decoder(predicted_selfie)
#     gen_smiles_cl3.append(predicted_smile)
#     gen_mols_cl3.append(Chem.MolFromSmiles(predicted_smile))

# # Step 3.2.1: Draw pictures of the compounds
# count = 0
# for gen_mol_cl3 in gen_mols_cl3:
#     img = Draw.MolToImage(gen_mol_cl3)
#     img_path = f'imgs/cl3/cl3_ligand_{count}' 
#     img.save(img_path)
#     count += 1

# # Step 3.3: Check uniqueness, novelty, SA, QED
# unique_smiles_cl3_set = set(gen_smiles_cl3)
# unique_smiles_cl3_list = list(unique_smiles_cl3_set)

# print(f"Uniqueness: {len(unique_smiles_cl3_list)} / {len(gen_smiles_cl3)}")



# Step 3.4: Run a docking simulation and check binding energies

