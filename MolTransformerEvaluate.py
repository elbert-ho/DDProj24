import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from MolTransformer import MultiTaskTransformer
from MolLoader import SMILESDataset

def evaluate_model(model, val_loader, tokenizer, max_length, device):
    model.eval()
    original_sequences = []
    predicted_sequences = []
    true_values = [[] for _ in range(5)]
    predicted_values = [[] for _ in range(5)]
    correct_symbols = 0
    total_symbols = 0

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            # if(batch_num == 8):
                # break
            src, task_targets = batch
            src = src.to(device)
            task_targets = task_targets.to(device)

            # Encode the input SMILES to get the representation for each token
            token_representations, fingerprints = model.encode_smiles(src)

            # Decode each token representation to get the predicted SMILES
            decoded_sequences, task_outputs = model.decode_representation(token_representations, fingerprints, max_length, tokenizer)

            src = src.cpu().numpy()
            decoded_sequences = decoded_sequences.cpu().numpy()
            task_targets = task_targets.cpu().numpy()

            # Ensure task outputs are correctly formatted and moved to CPU
            task_outputs = task_outputs.cpu().numpy()

            for j in range(src.shape[0]):
                original_seq = src[j, :]
                predicted_seq = decoded_sequences[j, :]

                # Convert token ids back to SMILES strings
                original_smiles = tokenizer.decode(original_seq.tolist(), skip_special_tokens=True)
                predicted_smiles = tokenizer.decode(predicted_seq.tolist(), skip_special_tokens=True)

                original_sequences.append(original_smiles)
                predicted_sequences.append(predicted_smiles)

                for k in range(5):
                    true_values[k].append(task_targets[j, k])
                    predicted_values[k].append(task_outputs[j][k])

                # Calculate correct symbols
                for orig_token, pred_token in zip(original_seq, predicted_seq):
                    if orig_token == tokenizer.token_to_id("[EOS]"):
                        break
                    if orig_token != tokenizer.token_to_id("[PAD]"):
                        total_symbols += 1
                        if orig_token == pred_token:
                            correct_symbols += 1

    # Prepare the data for the CSV file
    data = {
        'Original_SMILES': original_sequences,
        'Predicted_SMILES': predicted_sequences
    }
    for k in range(5):
        data[f'True_Value_{k+1}'] = true_values[k]
        data[f'Predicted_Value_{k+1}'] = predicted_values[k]

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv('data/predicted_smiles_transformer.csv', index=False)

    # Calculate and print percentage of correct symbols
    if total_symbols > 0:
        accuracy = correct_symbols / total_symbols * 100
        print(f'Percentage of correct symbols: {accuracy:.2f}%')
    else:
        print('No symbols to evaluate.')

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src_vocab_size = 1000
tgt_vocab_size = 1000
d_model = 256  # Updated d_model to 128
num_heads = 16
num_layers = 2
d_ff = 4000
max_seq_length = 128
dropout = 0.13
batch_size = 16
learning_rate = 1e-4
patience = 7  # Early stopping patience
reconstruction_loss_weight = 5.0  # Emphasis on reconstruction loss
num_tasks=5

model = MultiTaskTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks)
model.load_state_dict(torch.load('models/best_model.pt'))

# Ensure the model is on the correct device
model.to(device)

# Initialize the dataset and dataloaders
file_path = "data/smiles_10000_with_props.csv"
smiles_tokenizer = Tokenizer.from_file('models/smiles_tokenizer.json')
dataset = SMILESDataset(file_path, vocab_size=1000, max_length=128, tokenizer=smiles_tokenizer)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Evaluate model on the validation set
evaluate_model(model, val_loader, dataset.tokenizer, max_seq_length, device)
