import torch
from DiffusionModelGLIDE import *
import yaml
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, Subset
from unet_condition import Text2ImUNet
from ProtLigDataset import ProtLigDataset
from MolTransformerSelfies import MultiTaskTransformer
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem, Descriptors, QED
from rdkit.DataStructs import FingerprintSimilarity
from SelfiesTok import SelfiesTok
import numpy as np
import pandas as pd
import random
import umap
import matplotlib.pyplot as plt
from TestingUtils import *
import sys
import os
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'share', 'RDKit', 'Contrib', 'SA_Score'))
import sascorer
import selfies as sf
import argparse
from ESM2Regressor import ESM2Regressor
from transformers import EsmTokenizer, EsmModel

use_amp = False
cfg_fine = True

if(use_amp):
    scaler = torch.cuda.amp.GradScaler()
torch.manual_seed(40)

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
# epochs = config["diffusion_model"]["epochs"]
patience = config["diffusion_model"]["patience"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diffusion_model = GaussianDiffusion(betas=get_named_beta_schedule(n_diff_step))
unet = Text2ImUNet(text_ctx=1, xf_width=protein_embedding_dim, xf_layers=0, xf_heads=0, xf_final_ln=0, tokenizer=None, in_channels=256, model_channels=256, out_channels=512, num_res_blocks=2, attention_resolutions=[], dropout=.1, channel_mult=(1, 2, 4, 8), dims=1)
unet.load_state_dict(torch.load('unet_resized_even.pt', map_location=device))
unet.to(device)

class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, initial_molecules, diffusion_model, unet, protein_finger, mol_model, max_seq_length, d_model, max_steps=1000, num_samples=1000):
        self.molecules = initial_molecules  # List of 50 molecules as 256x128 arrays
        self.diffusion_model = diffusion_model
        self.unet = unet
        self.protein_finger = protein_finger
        self.mol_model = mol_model  # Pass mol_model for decoding representations
        self.max_seq_length = max_seq_length  # Sequence length for the molecular model
        self.d_model = d_model  # Dimension of the molecular model
        self.max_steps = max_steps
        self.num_samples = num_samples
        self.dataset = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Generate the artificial dataset
        self.generate_dataset()

    def generate_dataset(self):
        for molecule in self.molecules:
            self.dataset.append(molecule.cpu().numpy())  # Add the original 256 x 128 molecules
        # First, decode the original 50 molecules to SMILES strings to use for comparison
        original_smiles = self.get_original_smiles()

        full_smiles = []
        for molecule in original_smiles:
            full_smiles.append(molecule)

        while len(self.dataset) < self.num_samples:
            step = 200
            time = torch.tensor([step], device=self.device).repeat(len(self.molecules), 1)
            reference_sample = self.molecules.to(self.device)
            img = self.diffusion_model.q_sample(reference_sample, time)

            shape = (len(self.molecules), 256, 128)
            indices = list(range(step))[::-1]
            indices = tqdm(indices)

            for i in indices:
                t = torch.tensor([i] * shape[0], device=self.device)
                with torch.no_grad():
                    out = self.diffusion_model.p_sample(
                        self.unet,
                        img,
                        t,
                        prot=self.protein_finger,
                        w=5,
                        clip_denoised=True,
                        denoised_fn=None,
                        cond_fn=None,
                        model_kwargs=None,
                    )
                    img = out["sample"]

            # Reshape the generated sample
            sample = img.reshape(-1, 256, 128)

            # Prepare SMILES tokenizer
            tokenizer = SelfiesTok.load("models/selfies_tok.json")

            # Rescale and decode the generated molecules
            mins = torch.tensor(np.load("data/smiles_mins_selfies.npy"), device=self.device).reshape(1, 1, -1)
            maxes = torch.tensor(np.load("data/smiles_maxes_selfies.npy"), device=self.device).reshape(1, 1, -1)
            sample_rescale = (((sample.reshape(-1, 1, 32768) + 1) / 2) * (maxes - mins) + mins)

            decoded_smiles, _ = self.mol_model.decode_representation(
                sample_rescale.reshape(-1, self.max_seq_length, self.d_model),
                None,
                max_length=128,
                tokenizer=tokenizer
            )

            # Iterate over the generated samples, calculate similarity, and filter based on threshold
            for idx, molecule_vector in enumerate(sample):
                predicted_selfie = tokenizer.decode(decoded_smiles[idx].detach().cpu().flatten().tolist(), skip_special_tokens=True)
                predicted_smile = sf.decoder(predicted_selfie)

                try:
                    mol = Chem.MolFromSmiles(predicted_smile)
                    if mol:
                        # Get the original SMILES for this molecule (based on its index in the original 50)
                        original_mol = Chem.MolFromSmiles(original_smiles[idx])
                        if original_mol:
                            original_fp = AllChem.GetMorganFingerprintAsBitVect(original_mol, radius=2, nBits=2048)
                            generated_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

                            similarity = FingerprintSimilarity(original_fp, generated_fp)

                            # Only keep molecules with similarity between 0.15 and 0.7
                            if 0.3 <= similarity <= 0.8:
                                full_smiles.append(predicted_smile)
                                # Add the valid 256 x 128 representation to the dataset
                                self.dataset.append(molecule_vector.cpu().numpy())

                except Exception as e:
                    continue
            
            print(f"Dataset length: {len(self.dataset)}")
            print(np.mean(calculate_internal_pairwise_similarities(full_smiles)))

            # Stop if the dataset reaches the desired size
            if len(self.dataset) >= self.num_samples:
                break

        # print(np.mean(calculate_internal_pairwise_similarities(full_smiles)))
        # exit()

    def get_original_smiles(self):
        """
        Decode the original 50 molecules to get their SMILES representations.
        This will be used to compare the generated molecules against their respective original molecule.
        """
        tokenizer = SelfiesTok.load("models/selfies_tok.json")
        original_smiles = []
        for molecule_vector in self.molecules:
            # Rescale and decode the original molecules using the mol_model
            molecule_vector = molecule_vector.reshape(1, 1, -1)
            mins = torch.tensor(np.load("data/smiles_mins_selfies.npy"), device=self.device).reshape(1, 1, -1)
            maxes = torch.tensor(np.load("data/smiles_maxes_selfies.npy"), device=self.device).reshape(1, 1, -1)
            molecule_rescale = (((molecule_vector + 1) / 2) * (maxes - mins) + mins).to(self.device)

            decoded_smiles, _ = self.mol_model.decode_representation(
                molecule_rescale.reshape(1, self.max_seq_length, self.d_model),
                None,
                max_length=128,
                tokenizer=tokenizer
            )

            # Decode the SMILES string from the SELFIES representation
            predicted_selfie = tokenizer.decode(decoded_smiles[0].detach().cpu().flatten().tolist(), skip_special_tokens=True)
            predicted_smile = sf.decoder(predicted_selfie)
            original_smiles.append(predicted_smile)

        return original_smiles

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx], dtype=torch.float32), protein_finger[0]


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
unet_orig = Text2ImUNet(text_ctx=1, xf_width=protein_embedding_dim, xf_layers=0, xf_heads=0, xf_final_ln=0, tokenizer=None, in_channels=256, model_channels=256, out_channels=512, num_res_blocks=2, attention_resolutions=[], dropout=.1, channel_mult=(1, 2, 4, 8), dims=1)
mol_model = MultiTaskTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks).to(device)

unet_orig.load_state_dict(torch.load('unet_resized_even-9850.pt', map_location=device))
mol_model.load_state_dict(torch.load('models/selfies_transformer_final.pt', map_location=device))

unet_orig, mol_model = unet_orig.to(device), mol_model.to(device)
orig_smiles = torch.tensor(np.load("data/smiles_output_selfies_normal.npy")[9653:9703], device=device).reshape(50, 256, 128)

protein_sequence = "MSGPRAGFYRQELNKTVWEVPQRLQGLRPVGSGAYGSVCSAYDARLRQKVAVKKLSRPFQSLIHARRTYRELRLLKHLKHENVIGLLDVFTPATSIEDFSEVYLVTTLMGADLNNIVKCQALSDEHVQFLVYQLLRGLKYIHSAGIIHRDLKPSNVAVNEDCELRILDFGLARQADEEMTGYVATRWYRAPEIMLNWMHYNQTVDIWSVGCIMAELLQGKALFPGSDYIDQLKRIMEVVGTPSPEVLAKISSEHARTYIQSLPPMPQKDLSSIFRGANPLAIDLLGRMLVLDSDQRVSAAEALAHAYFSQYHDPEDEPEAEPYDESVEAKERTLEEWKELTYQEVLSFKPPEPPKPPGSLEIEQ"
protein_model_name = "facebook/esm2_t6_8M_UR50D"
protein_tokenizer = EsmTokenizer.from_pretrained(protein_model_name)
protein_model = EsmModel.from_pretrained(protein_model_name).to('cuda')
encoded_protein = protein_tokenizer(protein_sequence, return_tensors='pt', padding=True, truncation=True).to('cuda')
# Generate protein embeddings
with torch.no_grad():
    protein_outputs = protein_model(**encoded_protein)
    protein_embeddings = protein_outputs.last_hidden_state

    # representation = model_reg.get_rep(protein_sequence).flatten().detach().cpu().numpy()
    # Mean and Max Pooling
    mean_pooled = protein_embeddings.mean(dim=1)
    # max_pooled = protein_embeddings.max(dim=1).values
    # combined_pooled = torch.cat((mean_pooled, max_pooled), dim=1)
    combined_pooled = mean_pooled
protein_embedding = combined_pooled
protein_finger1 = torch.tensor(protein_embedding.reshape(1, -1), device=device)
protein_finger = protein_finger1.repeat(50, 1)

dataset = MoleculeDataset(orig_smiles, diffusion_model, unet_orig, protein_finger, mol_model, max_seq_length, d_model, num_samples=1000)

# dataset = ProtLigDataset('data/protein_embeddings2.npy', 'data/smiles_output_selfies_normal2.npy', 'data/protein_embeddings.npy', 'data/smiles_output_selfies_normal.npy')
# dataset = ProtLigDataset('data/protein_representations2.npy', 'data/smiles_output_selfies_normal2.npy', 'data/protein_representations.npy', 'data/smiles_output_selfies_normal.npy')

# train_size = int(1 * len(dataset))
# val_size = len(dataset) - train_size
# generator1 = torch.Generator().manual_seed(42)
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator1)

# Create indices for the train and test splits
# test_indices = list(range(0))  # First 10 samples
# train_indices = list(range(20, len(dataset)))  # Remaining samples

# Create subsets using the indices
# train_dataset = Subset(dataset, train_indices)
# val_dataset = Subset(dataset, test_indices)


# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

optimizer = optim.AdamW(unet.parameters(), lr=lr)
best_val_loss = float('inf')
patience_counter = 0

epochs = 1000
for epoch in range(epochs):
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
    #             prot = torch.zeros((b, protein_embedding_dim), device=device)

    #         t = torch.randint(0, n_diff_step, [b,] ,device=device)
    #         # t = torch.tensor([1], device=device).repeat(b)
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
    
    # exit()

    unet.train()
    epoch_loss = 0
    train_mse_loss = 0
    train_vb_loss = 0
    train_grad_norm = 0
    for mol, prot in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
        mol = mol.to(device)
        prot = prot.to(device)
        b = mol.shape[0]

        if(torch.randint(0, 5, (1,))[0] == 0):
            prot = torch.zeros((b, protein_embedding_dim), device=device)

        t = torch.randint(0, n_diff_step, [b,] ,device=device)
        # t = torch.randint(0, 100, [b,] ,device=device)
        # t = torch.tensor([1], device=device).repeat(b)
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
            
            
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1)
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
    if epoch % 100 == 0:
        if epoch % 2 == 0:
            torch.save(unet.state_dict(), 'unet_resized_even.pt')
        else:
            torch.save(unet.state_dict(), 'unet_resized_odd.pt')

torch.save(unet.state_dict(), 'unet_resized_even.pt')
