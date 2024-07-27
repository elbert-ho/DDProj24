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
import yaml
from tqdm import tqdm


def mean_flat(tensor):
    return torch.mean(tensor.view(tensor.size(0), -1), dim=1)

def train_model(dataset, epochs=10, batch_size=32, lr=1e-4, lambda_vlb=0.001, num_diffusion_steps=1000, diffusion_model=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(epochs):
        print(f"EPOCH {epoch + 1} BEGIN")
        epoch_loss = 0
        batch_count = 0
        
        for original_molecule, protein_embedding, _, _ in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
            original_molecule = original_molecule.to(device)
            protein_embedding = protein_embedding.to(device)
            batch_size = original_molecule.size(0)
            
            optimizer.zero_grad()
            
            noise = torch.randn_like(original_molecule)
            vb = []
            xstart_mse = []
            mse = []

            for t in reversed(range(num_diffusion_steps)):
                t_batch = torch.tensor([t] * batch_size, device=device)
                noised_molecule = diffusion_model.noise_molecule(original_molecule, t_batch, noise=noise)
                
                with torch.no_grad():
                    out = diffusion_model._vb_terms_bpd(
                        original_molecule=original_molecule,
                        noised_molecule=noised_molecule,
                        time_step=t_batch,
                        protein_embedding=protein_embedding
                    )
                
                vb.append(out["output"])
                xstart_mse.append(mean_flat((out["pred_xstart"] - original_molecule) ** 2))
                eps = diffusion_model._predict_eps_from_xstart(noised_molecule, t_batch, out["pred_xstart"])
                mse.append(mean_flat((eps - noise) ** 2))
            
            vb = torch.stack(vb, dim=1)
            xstart_mse = torch.stack(xstart_mse, dim=1)
            mse = torch.stack(mse, dim=1)
            
            # Compute L_simple
            L_simple = mse.mean()

            # Compute L_vlb
            L_vlb = vb.mean()

            # Compute L_hybrid
            L_hybrid = L_simple + lambda_vlb * L_vlb

            L_hybrid.backward()
            optimizer.step()

            epoch_loss += L_hybrid.item()
            batch_count += 1

        average_loss = epoch_loss / batch_count
        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {average_loss:.4f}")

    return diffusion_model

with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

protein_embedding_dim = config["protein_model"]["protein_embedding_dim"]
num_diffusion_steps = config["diffusion_model"]["num_diffusion_steps"]
batch_size = config["diffusion_model"]["batch_size"]
lr = config["diffusion_model"]["lr"]
epochs = config["diffusion_model"]["epochs"]
patience = config["diffusion_model"]["patience"]
lambda_vlb = config["diffusion_model"]["lambda_vlb"]
unet_ted = config["UNet"]["time_embedding_dim"]
device = config["mol_model"]["device"]

# Initialize models
unet_model = UNet1D(input_channels=1, output_channels=1, time_embedding_dim=unet_ted, protein_embedding_dim=protein_embedding_dim, num_diffusion_steps=num_diffusion_steps, device=device).to(device)

# Assuming original_molecule data is available in the dataset
diffusion_model = DiffusionModel(unet_model=unet_model, num_diffusion_steps=num_diffusion_steps, device=device)
dataset = pIC50Dataset('data/protein_embeddings.npy', 'data/smiles_output.npy', 'data/pic50.npy', diffusion_model=diffusion_model, num_diffusion_steps=num_diffusion_steps)
diffusion_model = diffusion_model.to(device)

# Train the model
print("BEGIN TRAIN")
trained_diffusion_model = train_model(dataset, epochs=epochs, batch_size=batch_size, lr=lr, lambda_vlb=lambda_vlb, num_diffusion_steps=num_diffusion_steps, diffusion_model=diffusion_model)
torch.save(unet_model.state_dict(), 'models/unet_model.pt')
