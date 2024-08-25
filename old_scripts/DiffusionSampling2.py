import torch
from pIC50Predictor import pIC50Predictor
from MolPropModel import PropertyModel
from MolTransformer import MultiTaskTransformer
from DiffusionModel import DiffusionModel
from tokenizers import Tokenizer
from UNet import UNet1D
import yaml
import numpy as np
import gc
import pandas as pd
from Bio import SeqIO
from rdkit import Chem

# Function to check validity of SMILES
def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

# Function to sample SMILES strings
def sample(diffusion_model, pic50_model, qed_model, sas_model, mol_transformer, tokenizer, protein_embedding, num_steps, gradient_scale, device, molecule_dim, max_seq_length, d_model):
    scaler = torch.cuda.amp.GradScaler()
    x = torch.randn(1, 1, molecule_dim, device=device, requires_grad=True)
    eps = 1e-8

    for t in range(num_steps - 1, -1, -1):
        time_step = torch.tensor([t], device=device)
        protein_embedding_squeeze = protein_embedding.unsqueeze(1).to(device)
        time_step = time_step.unsqueeze(0)

        with torch.cuda.amp.autocast():
            g_t = torch.zeros(3, *x.shape, device=device)  # Smoothed gradients
            log_var, epsilon_pred = diffusion_model(x, time_step, protein_embedding_squeeze)

            alpha_t = diffusion_model.alpha[t]
            alpha_bar_t = diffusion_model.alpha_bar[t]

            # Transform epsilon back to mu
            mu = (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_t)

            # Compute gradients from the pIC50, QED, and SAS models
            pic50_pred = pic50_model(x, protein_embedding_squeeze, time_step)
            qed_pred = qed_model(x, time_step)
            sas_pred = sas_model(x, time_step)

            g_t[0] = torch.autograd.grad(pic50_pred, x)[0]
            g_t[1] = torch.autograd.grad(qed_pred, x)[0]
            g_t[2] = -torch.autograd.grad(sas_pred, x)[0]

            g_norms = torch.linalg.vector_norm(g_t, dim=1)
            weights = torch.max(g_norms) / (g_norms + eps).unsqueeze(1)

            g_k = torch.sum(weights * g_t, dim=0)
            sigma = torch.sqrt(diffusion_model.beta_schedule[t])

            x = mu + sigma * gradient_scale * g_k
            x = x + sigma * torch.randn_like(x)

            x = x.detach().requires_grad_(True)
        # Clear CUDA cache to free up memory
        del log_var, epsilon_pred, pic50_pred, qed_pred, sas_pred, protein_embedding_squeeze, g_t, g_k, sigma
        torch.cuda.empty_cache()
        gc.collect()

    # Decode to SMILES
    with torch.no_grad():
        x = x.reshape(1, max_seq_length, d_model)
        decoded_smiles, _ = mol_transformer.decode_representation(x, None, max_length=100, tokenizer=tokenizer)
    
    return decoded_smiles

# Sample multiple SMILES for each protein
def sample_multiple_smiles(num_samples, diffusion_model, pic50_model, qed_model, sas_model, mol_transformer, tokenizer, protein_embedding, num_steps, gradient_scale, device, molecule_dim, max_seq_length, d_model):
    smiles_list = []
    for _ in range(num_samples):
        sampled_smiles = sample(diffusion_model, pic50_model, qed_model, sas_model, mol_transformer, tokenizer, protein_embedding, num_steps, gradient_scale, device, molecule_dim, max_seq_length, d_model)
        smiles_str = tokenizer.decode(sampled_smiles.detach().cpu().flatten().tolist(), skip_special_tokens=True)
        smiles_list.append(smiles_str)
    return smiles_list

# Evaluate validity, uniqueness, and novelty
def evaluate_smiles(smiles_list, known_smiles_set):
    valid_smiles = [smiles for smiles in smiles_list if is_valid_smiles(smiles)]
    unique_smiles = set(valid_smiles)
    novel_smiles = [smiles for smiles in unique_smiles if smiles not in known_smiles_set]

    num_valid = len(valid_smiles)
    num_unique = len(unique_smiles)
    num_novel = len(novel_smiles)

    total = len(smiles_list)
    valid_percentage = (num_valid / total) * 100
    unique_percentage = (num_unique / total) * 100
    novel_percentage = (num_novel / total) * 100

    return valid_percentage, unique_percentage, novel_percentage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

protein_embedding_dim = config["protein_model"]["protein_embedding_dim"]
num_diffusion_steps = config["diffusion_model"]["num_diffusion_steps"]
batch_size = config["diffusion_model"]["batch_size"]
epochs = config["diffusion_model"]["epochs"]
patience = config["diffusion_model"]["patience"]
lambda_vlb = config["diffusion_model"]["lambda_vlb"]
molecule_dim = config["mol_model"]["d_model"] * config["mol_model"]["max_seq_length"]
pIC_time_embed_dim = config["pIC50_model"]["time_embed_dim"]
pIC_hidden_dim = config["pIC50_model"]["hidden_dim"]
pIC_num_heads = config["pIC50_model"]["num_heads"]
prop_time_embed_dim = config["prop_model"]["time_embed_dim"]
src_vocab_size = config["mol_model"]["src_vocab_size"]
tgt_vocab_size = config["mol_model"]["tgt_vocab_size"]
max_seq_length = config["mol_model"]["max_seq_length"]
num_tasks = config["mol_model"]["num_tasks"]
d_model = config["mol_model"]["d_model"]
num_heads = config["mol_model"]["num_heads"]
num_layers = config["mol_model"]["num_layers"]
d_ff = config["mol_model"]["d_ff"]
dropout = config["mol_model"]["dropout"]
tok_file = config["mol_model"]["tokenizer_file"]
unet_ted = config["UNet"]["time_embedding_dim"]
gradient_scale = config["diffusion_model"]["gradient_scale"]

pretrained_pic50_model = pIC50Predictor(molecule_dim, protein_embedding_dim, pIC_hidden_dim, pIC_num_heads, pIC_time_embed_dim, num_diffusion_steps).to(device)
pretrained_qed_model = PropertyModel(molecule_dim, num_diffusion_steps, prop_time_embed_dim).to(device)
pretrained_sas_model = PropertyModel(molecule_dim, num_diffusion_steps, prop_time_embed_dim).to(device)
trained_diffusion_model = UNet1D(input_channels=1, output_channels=1, time_embedding_dim=unet_ted, protein_embedding_dim=protein_embedding_dim, num_diffusion_steps=num_diffusion_steps).to(device)

pretrained_pic50_model.load_state_dict(torch.load('models/pIC50_model.pt', map_location=device))
pretrained_qed_model.load_state_dict(torch.load('models/qed_model.pt', map_location=device))
pretrained_sas_model.load_state_dict(torch.load('models/sas_model.pt', map_location=device))
trained_diffusion_model.load_state_dict(torch.load('models/unet_model_no_var.pt', map_location=device))

diffusion_model = DiffusionModel(unet_model=trained_diffusion_model, num_diffusion_steps=num_diffusion_steps, device=device).to(device)

mol_model = MultiTaskTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks).to(device)
mol_model.load_state_dict(torch.load('models/pretrain.pt', map_location=device))
tokenizer = Tokenizer.from_file(tok_file)

# Load known SMILES (you can adjust this based on your known set)
known_smiles_set = set()  # Add known SMILES strings to this set

# Load protein embeddings
proteins = np.load('data/protein_embeddings.npy')

# Prepare the CSV output
output_csv = 'path/to/your/output_file.csv'
csv_rows = []

# Iterate through proteins and sample SMILES
for idx, protein_embedding in enumerate(proteins):
    protein_embedding = torch.FloatTensor(protein_embedding).to(device)
    smiles_list = sample_multiple_smiles(10, diffusion_model, pretrained_pic50_model, pretrained_qed_model, pretrained_sas_model, mol_model, tokenizer, protein_embedding, num_steps=num_diffusion_steps, gradient_scale=gradient_scale, device=device, molecule_dim=molecule_dim, max_seq_length=max_seq_length, d_model=d_model)

    for smiles in smiles_list:
        csv_rows.append({
            "Protein Sequence": f"Protein_{idx}",
            "SMILES String": smiles
        })

# Write the CSV file
csv_df = pd.DataFrame(csv_rows)
csv_df.to_csv(output_csv, index=False)

# Evaluate the SMILES
valid_percentage, unique_percentage, novel_percentage = evaluate_smiles([row["SMILES String"] for row in csv_rows], known_smiles_set)

print(f"Valid SMILES: {valid_percentage:.2f}%")
print(f"Unique SMILES: {unique_percentage:.2f}%")
print(f"Novel SMILES: {novel_percentage:.2f}%")
