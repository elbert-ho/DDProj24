import torch
from DiffusionModelGLIDE import *
import yaml
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from unet_condition import Text2ImUNet
from ProtLigDataset import ProtLigDataset

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)



with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

n_diff_step = config["diffusion_model"]["num_diffusion_steps"]
batch_size = config["diffusion_model"]["batch_size"]
protein_embedding_dim = config["protein_model"]["protein_embedding_dim"]
lr = config["diffusion_model"]["lr"]
epochs = config["diffusion_model"]["epochs"]
patience = config["diffusion_model"]["patience"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

diffusion_model = GaussianDiffusion(betas=get_named_beta_schedule(n_diff_step))
unet = Text2ImUNet(text_ctx=1, xf_width=protein_embedding_dim, xf_layers=0, xf_heads=0, xf_final_ln=0, tokenizer=None, in_channels=1, model_channels=48, out_channels=2, num_res_blocks=2, attention_resolutions=[], dropout=.1, channel_mult=(1, 2, 4, 8), dims=1)
unet.to(device)

dataset = ProtLigDataset('data/protein_embeddings.npy', 'data/smiles_output_normal.npy')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

optimizer = optim.Adam(unet.parameters(), lr=lr)
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    print(f"EPOCH {epoch + 1} BEGIN")
    
    unet.eval()
    val_loss = 0
    with torch.no_grad():
        for mol, prot in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{epochs}', leave=False):
            mol = mol.to(device).unsqueeze(1)
            prot = prot.to(device)
            b = mol.shape[0]
            t = torch.randint(0, n_diff_step, [b,] ,device=device)
            loss = torch.mean(diffusion_model.training_losses(unet, mol, t, prot=prot)["loss"])
            val_loss += loss.item()
    
    average_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Average Validation Loss: {average_val_loss:.4f}")
    
    unet.train()
    epoch_loss = 0
    for mol, prot in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
        mol = mol.to(device).unsqueeze(1)
        prot = prot.to(device)
        b = mol.shape[0]
        t = torch.randint(0, n_diff_step, [b,] ,device=device)
        optimizer.zero_grad()
        loss = torch.mean(diffusion_model.training_losses(unet, mol, t, prot=prot)["loss"])
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    average_train_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Average Training Loss: {average_train_loss:.4f}")

    # Early stopping
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        patience_counter = 0
        torch.save(unet.state_dict(), 'unet.pt')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break