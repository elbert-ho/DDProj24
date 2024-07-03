import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tokenizers import Tokenizer
from DiffusionModel import DiffusionModel
from pIC50Loader import pIC50Dataset
from UNet import UNet1D
from MolTransformer import MultiTaskTransformer
from pIC50Predictor import pIC50Predictor
from MolPropModel import PropertyModel


def train_model(unet_model, dataset, epochs=10, batch_size=32, lr=1e-4, lambda_vlb=1.0):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    diffusion_model = DiffusionModel(unet_model)
    
    optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)

    for epoch in range(epochs):
        for original_molecule, protein_embedding, time_step, _ in dataloader:
            optimizer.zero_grad()
            
            noise = torch.randn_like(original_molecule)
            noised_molecule = diffusion_model.noise_molecule(original_molecule, time_step, noise=noise)
            
            log_var, noise_pred = diffusion_model(noised_molecule, time_step, protein_embedding)
            
            # Compute L_simple
            L_simple = torch.mean((noise_pred - noise) ** 2)
            
            # Compute L_vlb
            q_mean = (noised_molecule - torch.sqrt(1 - diffusion_model.alpha_bar[time_step]) * noise) / torch.sqrt(diffusion_model.alpha_bar[time_step])
            L_vlb = 0.5 * torch.sum(log_var + (noised_molecule - q_mean) ** 2 / torch.exp(log_var) - 1, dim=1).mean()
            
            # Compute L_hybrid
            L_hybrid = L_simple + lambda_vlb * L_vlb
            
            L_hybrid.backward()
            optimizer.step()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {L_hybrid.item()}")

    return diffusion_model


# Initialize models
unet_model = UNet1D(input_channels=1, output_channels=1, time_embedding_dim=128, protein_embedding_dim=512)

# Assuming original_molecule data is available in the dataset
dataset = pIC50Dataset('protein_data.npy', 'smiles_data.npy', 'pic50_data.npy', num_diffusion_steps=1000)

# Train the model
trained_diffusion_model = train_model(unet_model, dataset)


def sample(diffusion_model, pic50_model, qed_model, sas_model, mol_transformer, tokenizer, protein_embedding, num_steps, gradient_scale, device):
    x = torch.randn(1, diffusion_model.input_channels, diffusion_model.image_size, device=device)
    g_t = torch.zeros(3, device=device)  # Gradient storage for pIC50, QED, and SAS
    g_t_hat = torch.zeros(3, device=device)  # Smoothed gradients
    g_t_hat_prev = torch.zeros(3, device=device)  # Previous smoothed gradients
    beta = 0.9
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
        
        g_t_hat_prev = g_t_hat.clone()
        g_t[0] = torch.autograd.grad(torch.log1p(pic50_pred.sum()), x, retain_graph=True)[0]
        g_t[1] = torch.autograd.grad(torch.log1p(qed_pred.sum()), x, retain_graph=True)[0]
        g_t[2] = -torch.autograd.grad(torch.log1p(sas_pred.sum()), x, retain_graph=True)[0]
        
        # Compute smoothed gradients
        g_t_hat = beta * g_t_hat_prev + (1 - beta) * g_t
        
        # Balance gradients using the balancing coefficient
        alpha_k = torch.max(torch.norm(g_t_hat, dim=1)) / (torch.norm(g_t_hat, dim=1) + eps)
        g_t_balanced = alpha_k[:, None] * g_t
        
        # Update x using the balanced gradient
        sigma = torch.exp(0.5 * log_var)
        combined_gradient = torch.mean(g_t_balanced, dim=0)  # Combine gradients
        x = mu + sigma * gradient_scale * combined_gradient
        x = x + sigma * torch.randn_like(x)
        
        g_t_hat_prev = g_t_hat

    # Decode to SMILES
    enc_output, fingerprint = mol_transformer.encode_smiles(x)
    decoded_smiles, _ = mol_transformer.decode_representation(enc_output, fingerprint, max_length=100, tokenizer=tokenizer)
    
    return decoded_smiles


# Example usage:
# trained_diffusion_model, pretrained_pic50_model, pretrained_qed_model, pretrained_sas_model, pretrained_mol_transformer, tokenizer

# Get the pretrained models
# Instantiate model, criterion, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = pIC50Predictor(16384, 1000, 1, 256, 8)

pretrained_pic50_model = pIC50Predictor()
pretrained_qed_model = PropertyModel(16384)
pretrained_sas_model = PropertyModel(16384)
pretrained_mol_transformer = MultiTaskTransformer(1000, 1000, 128, 8, 6, 2048, 128, 0.1, 5)

# Load the tokenizer
tokenizer = Tokenizer.from_file('models/smiles_tokenizer.json')

# Get the protein embedding
protein_embedding = None # TEMPORARY

# Call the sample function
sampled_smiles = sample(trained_diffusion_model, pretrained_pic50_model, pretrained_qed_model, pretrained_sas_model, pretrained_mol_transformer, tokenizer, protein_embedding, num_steps=1000, gradient_scale=1.0)

print("Sampled SMILES:", sampled_smiles)
