import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pandas as pd
import yaml
from tokenizers import Tokenizer
from MolTransformerSelfies import MultiTaskTransformer
from MolLoaderSelfies import SMILESDataset

def evaluate_model(model, val_loader, tokenizer, max_length, device):
    model.eval()
    original_sequences = []
    predicted_sequences = []
    true_values = [[] for _ in range(5)]
    predicted_values = [[] for _ in range(5)]
    correct_symbols = 0
    total_symbols = 0
    correct_seq = 0
    total_seq = 0

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            if(batch_num == 16):
                break
            src, task_targets = batch
            src = src.to(device)
            task_targets = task_targets.to(device)

            # Encode the input SMILES to get the representation for each token
            token_representations, fingerprints = model.encode_smiles(src)

            # print(src[0])
            # exit()

            # Decode each token representation to get the predicted SMILES
            decoded_sequences, task_outputs = model.decode_representation(token_representations, fingerprints, max_length, tokenizer)
            print(src[0])
            print(decoded_sequences[0])
            print(token_representations[0][0])
            exit()

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

                print(original_smiles)
                print(predicted_smiles)

                original_sequences.append(original_smiles)
                predicted_sequences.append(predicted_smiles)

                for k in range(5):
                    true_values[k].append(task_targets[j, k])
                    predicted_values[k].append(task_outputs[j][k])

                # Calculate correct symbols
                flag_correct = True
                for orig_token, pred_token in zip(original_seq, predicted_seq):
                    if orig_token == tokenizer.token_to_id["[EOS]"]:
                        break
                    if orig_token != tokenizer.token_to_id["[PAD]"]:
                        total_symbols += 1
                        if orig_token == pred_token:
                            correct_symbols += 1
                        else:
                            flag_correct = False
                
                total_seq += 1
                if(flag_correct):
                    correct_seq += 1


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
    df.to_csv('data/predicted_selfies_transformer.csv', index=False)

    # Calculate and print percentage of correct symbols
    if total_symbols > 0:
        accuracy = correct_symbols / total_symbols * 100
        print(f'Percentage of correct symbols: {accuracy:.2f}%')
    else:
        print('No symbols to evaluate.')

    if total_seq > 0:
        accuracy = correct_seq / total_seq * 100
        print(f'Percentage of correct sequences: {accuracy:.2f}%')
    else:
        print('No sequences to evaluate')

# Hyperparameters

with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

src_vocab_size = config["mol_model"]["src_vocab_size"]
tgt_vocab_size = config["mol_model"]["tgt_vocab_size"]
max_seq_length = config["mol_model"]["max_seq_length"]
num_tasks = config["mol_model"]["num_tasks"]
d_model = config["mol_model"]["d_model"]
num_heads = config["mol_model"]["num_heads"]
num_layers = config["mol_model"]["num_layers"]
d_ff = config["mol_model"]["d_ff"]
dropout = config["mol_model"]["dropout"]
learning_rate = config["mol_model"]["learning_rate"]
batch_size = config["mol_model"]["batch_size"]
device = config["mol_model"]["device"]
warmup_epochs = config["mol_model"]["warmup_epochs"]
total_epochs = config["mol_model"]["total_epochs"]
patience = config["mol_model"]["patience"]
pretrain_epochs = config["mol_model"]["pretrain_epochs"]
pretrain_learning_rate = config["mol_model"]["pretrain_learning_rate"]
tok_file = config["mol_model"]["tokenizer_file"]

model = MultiTaskTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks)
model.load_state_dict(torch.load('models/selfies_transformer2.pt'))

# Ensure the model is on the correct device
model.to(device)

# Initialize the dataset and dataloaders
file_path = "data/smiles_10000_selected_features_cleaned.csv"
dataset = SMILESDataset(file_path, vocab_size=1000, max_length=128, tokenizer_path="models/selfies_tok.json")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Evaluate model on the validation set
evaluate_model(model, val_loader, dataset.tokenizer, max_seq_length, device)
