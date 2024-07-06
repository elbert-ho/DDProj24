import torch
from pIC50Predictor import pIC50Predictor
from MolPropModel import PropertyModel
from MolTransformer import MultiTaskTransformer
from tokenizers import Tokenizer
from UNet import UNet1D
import yaml

def sample(diffusion_model, pic50_model, qed_model, sas_model, mol_transformer, tokenizer, protein_embedding, num_steps, gradient_scale, device):
    x = torch.randn(1, diffusion_model.input_channels, diffusion_model.image_size, device=device)
    # g_t = [None] * 3  # Gradient storage for pIC50, QED, and SAS
    g_t = torch.zeros(3, device=device)  # Smoothed gradients
    # g_t_hat_prev = torch.zeros(3, device=device)  # Previous smoothed gradients
    eps = 1e-8

    for t in range(num_steps, 0, -1):
        time_step = torch.tensor([t], device=device)
        log_var, epsilon_pred = diffusion_model(x, time_step, protein_embedding)
        
        alpha_t = diffusion_model.alpha[t]
        alpha_bar_t = diffusion_model.alpha_bar[t]
        
        # Transform epsilon back to mu
        mu = (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_t)
        
        # Compute gradients from the pIC50, QED, and SAS models
        pic50_pred = pic50_model(x, protein_embedding, time_step)
        
        qed_pred = qed_model(x)
        sas_pred = sas_model(x)
        
        # g_t_hat_prev = g_t_hat.clone()
        g_t[0] = torch.autograd.grad(pic50_pred, x)
        g_t[1] = torch.autograd.grad(qed_pred, x)
        g_t[2] = -torch.autograd.grad(sas_pred)

        g_norms = torch.linalg.vector_norm(g_t, dim=1)
        weights = torch.max(g_norms) / (g_norms + eps)
        g_k = torch.sum(weights * g_t, dim=0)
        sigma = torch.exp(0.5 * log_var)
        x = mu + sigma * gradient_scale * g_k
        x = x + sigma * torch.randn_like(x)

    # Decode to SMILES
    enc_output, fingerprint = mol_transformer.encode_smiles(x)
    decoded_smiles, _ = mol_transformer.decode_representation(enc_output, fingerprint, max_length=100, tokenizer=tokenizer)
    
    return decoded_smiles



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

pretrained_pic50_model = pIC50Predictor(molecule_dim, protein_embedding_dim, pIC_hidden_dim, pIC_num_heads, pIC_time_embed_dim, num_diffusion_steps)
pretrained_qed_model = PropertyModel(molecule_dim, num_diffusion_steps, prop_time_embed_dim)
pretrained_sas_model = PropertyModel(molecule_dim, num_diffusion_steps, prop_time_embed_dim)


mol_model = MultiTaskTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks)
mol_model.load_state_dict(torch.load('models/best_model.pt'))

# Load the tokenizer
tokenizer = Tokenizer.from_file(tok_file)

# Get the protein embedding
protein_embedding = None # TEMPORARY

trained_diffusion_model = UNet1D(input_channels=1, output_channels=1, time_embedding_dim=unet_ted, protein_embedding_dim=protein_embedding_dim, num_diffusion_steps=num_diffusion_steps)

# Call the sample function
# CHECK GRADIENT SCALE
sampled_smiles = sample(trained_diffusion_model, pretrained_pic50_model, pretrained_qed_model, pretrained_sas_model, mol_model, tokenizer, protein_embedding, num_steps=num_diffusion_steps, gradient_scale=1.0)

print("Sampled SMILES:", sampled_smiles)
