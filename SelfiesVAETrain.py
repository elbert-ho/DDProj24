import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from SelfiesVAE import VAE
from MolLoaderSelfies2 import *
import os

# =============================
# Loss Function
# =============================

def vae_loss(recon_x, x, mu, logvar, params):
    # Reshape recon_x to [batch_size * MAX_LEN, NCHARS]
    recon_x = recon_x.view(-1, recon_x.size(-1))  # Flatten recon_x to [batch_size * MAX_LEN, NCHARS]
    
    # Reshape x_target to [batch_size * MAX_LEN]
    x_target = torch.argmax(x, dim=2).view(-1)  # Flatten x to [batch_size * MAX_LEN]
    
    # Cross-entropy loss
    recon_loss = nn.CrossEntropyLoss(reduction='sum')(recon_x, x_target)
    
    # KL Divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD *= params['kl_loss_weight']
    
    return recon_loss + KLD

# =============================
# Training and Validation Functions
# =============================

def train_vae(model, dataloader, optimizer, device, params):
    model.train()
    total_loss = 0
    for _, batch in tqdm(dataloader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = vae_loss(recon_batch, batch, mu, logvar, params)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def validate_vae(model, dataloader, device, params):
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0
    total_sequences = 0
    completely_correct_sequences = 0
    with torch.no_grad():
        for _, batch in tqdm(dataloader, desc="Validation"):
            batch = batch.to(device)
            recon_batch, mu, logvar = model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar, params)
            total_loss += loss.item()

            # Convert recon_batch to token indices (argmax over the token dimension)
            recon_tokens = torch.argmax(recon_batch, dim=-1)  # Shape: [batch_size, MAX_LEN]
            batch_tokens = torch.argmax(batch, dim=-1)
            
            # Ensure batch has the correct shape for comparison
            # batch_tokens = batch  # Assuming `batch` is tokenized with shape [batch_size, MAX_LEN]

            # Calculate correct tokens by comparing recon_tokens and batch_tokens
            # print(batch_tokens.shape)
            # print(recon_tokens.shape)

            token_correct = torch.eq(recon_tokens, batch_tokens).float()  # [batch_size, MAX_LEN]
            correct_tokens += torch.sum(token_correct).item()
            total_tokens += torch.numel(batch_tokens)

            # Calculate whether entire sequences are correct (i.e., all tokens match)
            sequence_correct = torch.all(torch.eq(recon_tokens, batch_tokens), dim=-1).float()  # [batch_size]
            completely_correct_sequences += torch.sum(sequence_correct).item()
            total_sequences += batch.size(0)

    token_accuracy = (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    sequence_accuracy = (completely_correct_sequences / total_sequences) * 100 if total_sequences > 0 else 0

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, token_accuracy, sequence_accuracy

# =============================
# Main Script
# =============================

if __name__ == "__main__":
    # Hyperparameters from your given parameters
    params = {
        # General parameters
        'batch_size': 32,
        'epochs': 4000,  # Set to desired number of epochs
        'val_split': 0.1,
        'loss': 'categorical_crossentropy',
        'RAND_SEED': 42,
        'MAX_LEN': 128,  # Adjust based on your data
        'NCHARS': 79,  # Number of unique tokens in SMILES

        # Convolution parameters
        'batchnorm_conv': False,
        'conv_activation': 'tanh',
        'conv_depth': 4,
        'conv_dim_depth': 8,
        'conv_dim_width': 8,
        'conv_d_growth_factor': 1.15875438383,
        'conv_w_growth_factor': 1.1758149644,

        # Decoder parameters
        'gru_depth': 4,
        'rnn_activation': 'tanh',
        'recurrent_dim': 50,
        'do_tgru': True,
        'tgru_dropout': 0.0,
        'temperature': 1.00,

        # Middle layer parameters
        'hg_growth_factor': 1.4928245388,
        'hidden_dim': 100,
        'middle_layer': 1,
        'dropout_rate_mid': 0.0,
        'batchnorm_mid': False,
        'activation': 'tanh',

        # Optimization parameters
        'lr': 0.000312087049936,
        'momentum': 0.936948773087,
        'optim': 'adam',

        # VAE parameters
        'kl_loss_weight': 1.0,
    }

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed for reproducibility
    torch.manual_seed(params['RAND_SEED'])
    np.random.seed(params['RAND_SEED'])

    # Load your data
    # Create dataset and data loaders
    file_path = "data/smiles_10000_selected_features_cleaned.csv"
    dataset = SMILESDataset(file_path, vocab_size=params['NCHARS'], max_length=params['MAX_LEN'], tokenizer_path="models/selfies_tok.json")

    # Split dataset into training and validation
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(params['val_split'] * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=val_sampler)

    # Initialize the model, optimizer
    model = VAE(params).to(device)

    if params['optim'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    elif params['optim'] == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=params['lr'], momentum=params['momentum'])
    elif params['optim'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])
    else:
        raise ValueError("Unsupported optimizer type")

    checkpoint_name = "vae_model.pth"
    if(os.path.isfile(checkpoint_name)):
        checkpoint = torch.load(checkpoint_name)
        loaded_epochs = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        loaded_epochs = 0

    # Training loop
    for epoch in range(1, params['epochs'] + 1):
        train_loss = train_vae(model, train_loader, optimizer, device, params)
        val_loss, token_accuracy, sequence_accuracy = validate_vae(model, val_loader, device, params)
        print(f"Epoch {epoch}/{params['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Token Accuracy: {token_accuracy:.2f}, Sequence Accuracy: {sequence_accuracy:.2f}%")

    # Save the model checkpoint
    torch.save({
        'epoch': params['epochs'] + loaded_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'vae_model.pth')
