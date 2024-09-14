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
from rdkit.Chem.Scaffolds import MurckoScaffold
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
from pIC50Predictor2 import pIC50Predictor
import gc

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
unet = Text2ImUNet(text_ctx=1, xf_width=protein_embedding_dim, xf_layers=0, xf_heads=0, xf_final_ln=0, tokenizer=None, in_channels=256, model_channels=256, out_channels=512, num_res_blocks=2, attention_resolutions=[4], dropout=.1, channel_mult=(1, 2, 4, 8), dims=1)
mol_model = MultiTaskTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks).to(device)

unet.load_state_dict(torch.load('unet_resized_even_attention_4_tuned.pt', map_location=device))
mol_model.load_state_dict(torch.load('models/selfies_transformer_final.pt', map_location=device))

unet, mol_model = unet.to(device), mol_model.to(device)
mol_model.eval()

# Step 0.3: Pick 40 random proteins and generate 25 molecules per protein
# Group by Protein Sequence and filter groups that have at least 25 SMILES strings
ref_data_smiles.reset_index(inplace=True)
# protein_groups = ref_data_smiles.groupby('Protein Sequence').filter(lambda x: len(x) >= 50)

# Get unique protein sequences that have at least 25 SMILES strings
# unique_proteins = protein_groups['Protein Sequence'].unique()

# Select 40 random unique proteins HERE
# num_proteins = 2
# per_protein = 100

# random_proteins = random.sample(list(unique_proteins), num_proteins)

# For each selected protein, extract 25 random SMILES strings
# selected_rows = []
# for protein in random_proteins:
    # protein_df = protein_groups[protein_groups['Protein Sequence'] == protein]
    # selected_smiles = protein_df.sample(per_protein)
    # selected_rows.append(selected_smiles)

# ref_data_smiles_cut = pd.concat(selected_rows)

# Get ESM
# protein_indices = ref_data_smiles_cut.loc[:,"index"].to_numpy()
# protein_fingers_full = np.load("data/protein_embeddings.npy")
# protein_fingers_cut = []
# count = 0
# for protein_idx in protein_indices:
    # if count % per_protein == 0:
        # protein_fingers_cut.append(protein_fingers_full[protein_idx])
    # count += 1

# Step 0.3.1: Generate 25 molecules per protein
mins = torch.tensor(np.load("data/smiles_mins_selfies.npy"), device=device).reshape(1, 1, -1)
maxes = torch.tensor(np.load("data/smiles_maxes_selfies.npy"), device=device).reshape(1, 1, -1)
# ws = [0, 0.1, 0.3, 0.5, 1, 3, 5, 9]
ws = [5]
qed_scores_avgs = []
sas_scores_avgs = []
qed_percent_passed = []
sas_percent_passed = []
divs = []

per_protein = 100
protein_fingers_cut = []
# protein_sequences = ["SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQCSGVTFQ", 
#                       "MAHAGRTGYDNREIVMKYIHYKLSQRGYEWDAGDVGAAPPGAAPAPGIFSSQPGHTPHPAASRDPVARTSPLQTPAAPGAAAGPALSPVPPVVHLTLRQAGDDFSRRYRRDFAEMSSQLHLTPFTARGRFATVVEELFRDGVNWGRIVAFFEFGGVMCVESVNREMSPLVDNIALWMTEYLNRHLHTWIQDNGGWDAFVELYGPSMR", 
#                       "GSMSEQSICQARAAVMVYDDANKKWVPAGGSTGFSRVHIYHHTGNNTFRVVGRKIQDHQVVINCAIPKGLKYNQATQTFHQWRDARQVYGLNFGSKEDANVFASAMMHALEVL"]
protein_sequences = ["GSMSEQSICQARAAVMVYDDANKKWVPAGGSTGFSRVHIYHHTGNNTFRVVGRKIQDHQVVINCAIPKGLKYNQATQTFHQWRDARQVYGLNFGSKEDANVFASAMMHALEVL"]


protein_model_name = "facebook/esm2_t30_150M_UR50D"
protein_tokenizer = EsmTokenizer.from_pretrained(protein_model_name)
protein_model = EsmModel.from_pretrained(protein_model_name).to('cuda')
for protein_sequence in protein_sequences:
    encoded_protein = protein_tokenizer(protein_sequence, return_tensors='pt', padding=True, truncation=True).to('cuda')
    # Generate protein embeddings
    with torch.no_grad():
        protein_outputs = protein_model(**encoded_protein)
        protein_embeddings = protein_outputs.last_hidden_state

        # Mean and Max Pooling
        mean_pooled = protein_embeddings.mean(dim=1)
        max_pooled = protein_embeddings.max(dim=1).values
        combined_pooled = torch.cat((mean_pooled, max_pooled), dim=1)
    protein_embedding = combined_pooled.detach().cpu().numpy()
    protein_fingers_cut.append(protein_embedding.reshape(1, -1))

for w_c in ws:
    samples = torch.tensor([], device=device)
    for protein_finger in protein_fingers_cut:
        protein_finger = torch.tensor(protein_finger, device=device)
        protein_finger = protein_finger.repeat(per_protein, 1)
        sample = diffusion_model.p_sample_loop(unet, (per_protein, 256, 128), prot=protein_finger, w=w_c).detach().reshape(per_protein, 1, 32768)
        # sample = diffusion_model.p_sample_loop(unet, (10, 256, 128), prot=protein_finger, cond_fn=get_pIC50_grad).detach().reshape(per_protein, 1, 32768)
        samples = torch.cat([samples, sample])
    # Debug line
    # print(samples.shape)

    # Step 0.3.2: Remember to unnormalize
    sample_rescale = (((samples + 1) / 2) * (maxes - mins) + mins)
    # print(sample_rescale.shape)

    # Step 0.3.3: Convert from SELFIES back to SMILES
    tokenizer = SelfiesTok.load("models/selfies_tok.json")
    with torch.no_grad():
        decoded_smiles, _ = mol_model.decode_representation(sample_rescale.reshape(-1, max_seq_length, d_model), None, max_length=128, tokenizer=tokenizer)

    gen_smiles = []

    for decode in decoded_smiles:
        predicted_selfie = tokenizer.decode(decode.detach().cpu().flatten().tolist(), skip_special_tokens=True)
        predicted_smile = sf.decoder(predicted_selfie)
        gen_smiles.append(predicted_smile)

    # print(gen_smiles)

    total = len(gen_smiles)
    gen_sim_nonunique = calculate_internal_pairwise_similarities(gen_smiles)
    int_div_1 = 1 - (1 / (total * total)) * np.sum(gen_sim_nonunique)
    print(f"IntDiv1: {int_div_1}")
    divs.append(int_div_1)

    qeds = []
    sass = []
    passed_both = 0
    for smile in gen_smiles:
        try:
            mol = Chem.MolFromSmiles(smile)
            qed = QED.qed(mol)
            sas = sascorer.calculateScore(mol)
            qeds.append(qed)
            sass.append(sas)
            if qed >= .5 and sas <= 5:
                passed_both += 1
        except:
            pass

    qed_scores_avgs.append(sum(qeds)/len(qeds))
    sas_scores_avgs.append(sum(sass)/len(sass))
    qed_percent_passed.append(sum(1 for item in qeds if item >= .5) / len(qeds))
    sas_percent_passed.append(sum(1 for item in sass if item <= 5) / len(sass))

    print(passed_both / len(qeds))
    print(divs)
    print(qed_scores_avgs)
    print(sas_scores_avgs)
    print(qed_percent_passed)
    print(sas_percent_passed)
    gc.collect()

# plt.plot(ws, fcd_scores, 'o-', label='FCD Average vs. w')
total_averages = [a + (1 - b/10) for a, b in zip(qed_scores_avgs, sas_scores_avgs)]

plt.plot(total_averages, divs, 'o-', label='')
plt.xlabel('sum')
plt.ylabel('internal diversity')
plt.savefig("avg.png")

plt.clf()

plt.plot(qed_percent_passed, sas_percent_passed, 'o-', label='')
plt.xlabel('qed percent')
plt.ylabel('sas percent')
plt.savefig("percent.png")


