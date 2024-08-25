import torch
from DiffusionModelGLIDE import *
import yaml
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from unet_condition import Text2ImUNet
from ProtLigDataset import ProtLigDataset
use_amp = True
cfg_fine = True

if(use_amp):
    scaler = torch.cuda.amp.GradScaler()
torch.manual_seed(43)

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

def get_grad_norm(model):
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2.0).item() 

    return total_norm   

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
unet = Text2ImUNet(text_ctx=1, xf_width=protein_embedding_dim, xf_layers=0, xf_heads=0, xf_final_ln=0, tokenizer=None, in_channels=256, model_channels=256, out_channels=512, num_res_blocks=2, attention_resolutions=[4], dropout=.1, channel_mult=(1, 2, 4, 8), dims=1)
unet.load_state_dict(torch.load('unet_resized_even.pt', map_location=device))
unet.to(device)

dataset = ProtLigDataset('data/protein_embeddings2.npy', 'data/smiles_output_selfies_normal2.npy')
train_size = int(1 * len(dataset))
val_size = len(dataset) - train_size
generator1 = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator1)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

optimizer = optim.AdamW(unet.parameters(), lr=lr)
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(50):
    print(f"EPOCH {epoch + 1} BEGIN")
    
    # unet.eval()
    # val_loss = 0
    # val_mse_loss = 0
    # val_vb_loss = 0
    # with torch.no_grad():
    #     for mol, prot in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{epochs}', leave=False):
    #         mol = mol.to(device)
    #         prot = prot.to(device)
    #         b = mol.shape[0]
    #         # print(mol.shape)
    #         # exit()
    #         if(torch.randint(0, 5, (1,))[0] == 0):
    #             prot = torch.zeros((b, 1280), device=device)

    #         t = torch.randint(0, n_diff_step, [b,] ,device=device)
    #         loss_dict = diffusion_model.training_losses(unet, mol, t, prot=prot)
    #         loss = torch.mean(loss_dict["loss"])
    #         loss_mse = torch.mean(loss_dict["mse"])
    #         loss_vlb = torch.mean(loss_dict["vb"])
    #         val_loss += loss.item()
    #         val_mse_loss += loss_mse.item( )
    #         val_vb_loss += loss_vlb.item()
    
    # average_val_loss = val_loss / len(val_loader)
    # average_val_mse_loss = val_mse_loss / len(val_loader)
    # average_val_vb_loss = val_vb_loss / len(val_loader)
    # print(f"Epoch [{epoch + 1}/{epochs}], Average Validation Loss: {average_val_loss:.4f}, Average MSE Loss: {average_val_mse_loss}, Average VB Loss: {average_val_vb_loss}")
    
    unet.train()
    epoch_loss = 0
    train_mse_loss = 0
    train_vb_loss = 0
    train_grad_norm = 0
    for mol, prot in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
        mol = mol.to(device)
        prot = prot.to(device)
        b = mol.shape[0]
        t = torch.randint(0, n_diff_step, [b,] ,device=device)
        # t = torch.tensor([998], device=device).repeat(b)
        optimizer.zero_grad()

        if(use_amp):
            with torch.cuda.amp.autocast():
                # if(epoch == 9):
                    # loss_dict = diffusion_model.training_losses(unet, mol, t, prot=prot, debug=True)
                # else:
                    # loss_dict = diffusion_model.training_losses(unet, mol, t, prot=prot)
                loss_dict = diffusion_model.training_losses(unet, mol, t, prot=prot)
                loss = torch.mean(loss_dict["loss"])
                loss_mse = torch.mean(loss_dict["mse"])
                loss_vlb = torch.mean(loss_dict["vb"])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            loss_dict = diffusion_model.training_losses(unet, mol, t, prot=prot)
            loss = torch.mean(loss_dict["loss"])
            loss_mse = torch.mean(loss_dict["mse"])
            loss_vlb = torch.mean(loss_dict["vb"])
            loss.backward()
            optimizer.step()
            
            
        # torch.nn.utils.clip_grad_norm_(unet.parameters(), 1)
        # train_grad_norm += (get_grad_norm(unet))
        epoch_loss += loss.item()
        train_mse_loss += loss_mse.item()
        train_vb_loss += loss_vlb.item()

        # print(epoch_loss)
        # exit()
    
    # print(train_grad_norm / len(train_loader))
    average_train_loss = epoch_loss / len(train_loader)
    average_train_mse_loss = train_mse_loss / len(train_loader)
    average_train_vb_loss = train_vb_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Average Training Loss: {average_train_loss:.4f}, Average MSE Loss: {average_train_mse_loss}, Average VB Loss: {average_train_vb_loss}")

    # Early stopping
    # if average_val_loss < best_val_loss:
    #     best_val_loss = average_val_loss
    #     patience_counter = 0
    #     torch.save(unet.state_dict(), 'unet.pt')
    # else:
    #     patience_counter += 1

    # if patience_counter >= patience:
    #     print("Early stopping triggered")
    #     break
    if epoch % 5 == 0:
        if epoch % 2 == 0:
            torch.save(unet.state_dict(), 'unet_resized_even.pt')
        else:
            torch.save(unet.state_dict(), 'unet_resized_odd.pt')