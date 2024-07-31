import torch
from pIC50Predictor import pIC50Predictor
from MolPropModel import PropertyModel
from MolTransformer import MultiTaskTransformer
from DiffusionModel import DiffusionModel
from tokenizers import Tokenizer
from unet_condition import Text2ImUNet
import yaml
import numpy as np
import gc
from rdkit import Chem
from rdkit.Chem import Draw
from transformers import EsmTokenizer, EsmModel
import pandas as pd

def sample(diffusion_model, pic50_model, qed_model, sas_model, mol_transformer, tokenizer, protein_embedding, num_steps, gradient_scale, device, molecule_dim):
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
            qed_pred = qed_model(x, time_step).clamp(0, 1)
            sas_pred = sas_model(x, time_step).clamp(1, 10)

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
        decoded_smiles, _ = mol_transformer.decode_representation(x.reshape(1, max_seq_length, d_model), None, max_length=100, tokenizer=tokenizer)
    
    return decoded_smiles, x

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
trained_diffusion_model = Text2ImUNet(text_ctx=1, xf_width=protein_embedding_dim, xf_layers=0, xf_heads=0, xf_final_ln=0, tokenizer=None, in_channels=1, model_channels=48, out_channels=1, num_res_blocks=2, attention_resolutions=[], dropout=.1, channel_mult=(1, 2, 4, 8), dims=1)
diffusion_model = DiffusionModel(unet_model=trained_diffusion_model, num_diffusion_steps=num_diffusion_steps, device=device).to(device)

pretrained_pic50_model.load_state_dict(torch.load('models/pIC50_model.pt', map_location=device))
pretrained_qed_model.load_state_dict(torch.load('models/qed_model.pt', map_location=device))
pretrained_sas_model.load_state_dict(torch.load('models/sas_model.pt', map_location=device))
diffusion_model.load_state_dict(torch.load('best_diffusion_model.pt', map_location=device))

# diffusion_model = DiffusionModel(unet_model=trained_diffusion_model, num_diffusion_steps=num_diffusion_steps, device=device).to(device)
mol_model = MultiTaskTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks).to(device)
mol_model.load_state_dict(torch.load('models/pretrain.pt', map_location=device))
tokenizer = Tokenizer.from_file(tok_file)

proteins = np.load('data/protein_embeddings.npy')
protein_sequence = "SGLRKMAQPSGLVEPCIVRVSYGNNVLNGLWLGDEVICPRHVIASDTTRVINYENEMSSVRLHNFSVSKNNVFLGVVSARYKGVNLVLKVNQVNPNTPEHKFKSIKAGESFNILACYEGCPGSVYGVNMRSQGTIKGSFIAGTCGSVGYVLENGILYFVYMHHLELGNGSHVGSNFEGEMYGGYEDQPSMQLEGTNVMSSDNVVAFLYAALINGERWFVTNTSMSLESYNTWAKTNSFTELSSTDAFSMLAAKTGQSVEKLLDSIVRLNKGFGGRTILSYGSLCDEFTPTEVIRQMYGVNLQ"
protein_model_name = "facebook/esm2_t30_150M_UR50D"
protein_tokenizer = EsmTokenizer.from_pretrained(protein_model_name)
protein_model = EsmModel.from_pretrained(protein_model_name).to('cuda')
encoded_protein = protein_tokenizer(protein_sequence, return_tensors='pt', padding=True, truncation=True).to('cuda')
# Generate protein embeddings
with torch.no_grad():
    protein_outputs = protein_model(**encoded_protein)
    protein_embeddings = protein_outputs.last_hidden_state

    # Mean and Max Pooling
    mean_pooled = protein_embeddings.mean(dim=1)
    max_pooled = protein_embeddings.max(dim=1).values
    combined_pooled = torch.cat((mean_pooled, max_pooled), dim=1)
protein_embedding = combined_pooled

# Call the sample function
cur_df = pd.DataFrame(columns=["pIC50", "qed", "sas"])
valid = 0
for sample_idx in range(1000):
    sampled_smiles, final_x = sample(diffusion_model, pretrained_pic50_model, pretrained_qed_model, pretrained_sas_model, mol_model, tokenizer, protein_embedding, num_steps=num_diffusion_steps, gradient_scale=gradient_scale, device=device, molecule_dim=molecule_dim)
    print(sampled_smiles.detach().cpu().flatten())
    predicted_smiles = tokenizer.decode(sampled_smiles.detach().cpu().flatten().tolist(), skip_special_tokens=True)

    print("Sampled SMILES:", predicted_smiles)
    final_x = final_x.detach().to(device)
    protein_embedding = protein_embedding.to(device)
    pic50 = (pretrained_pic50_model(final_x, protein_embedding.unsqueeze(1), torch.tensor([1000], device=device).unsqueeze(0))).detach().cpu()
    qed = (pretrained_qed_model(final_x, torch.tensor([1000], device=device).unsqueeze(0)).clamp(0, 1)).detach().cpu()
    sas = (pretrained_sas_model(final_x, torch.tensor([1000], device=device).unsqueeze(0)).clamp(1, 10)).detach().cpu()

    predicted_smiles_str = str(predicted_smiles)
    try:
        mol = Chem.MolFromSmiles(predicted_smiles_str)
        img = Draw.MolToImage(mol)
        img_path = f'imgs/mol{sample_idx}.png' 
        img.save(img_path)
        valid += 1
    except:
        pass
    cur_df.loc[-1] = [pic50, qed, sas]
    cur_df.to_csv("diffused_mol.csv")
    print(f"VALID: {valid} / {sample_idx + 1}")
    


