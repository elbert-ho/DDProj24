import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from MolLoader import *
from MolTransformer import *

# Hyperparameters
src_vocab_size = 1000
tgt_vocab_size = 1000
d_model = 128  # Updated d_model to 128
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 128
dropout = 0.1
batch_size = 32
learning_rate = 1e-4
patience = 3  # Early stopping patience
num_tasks = 5  # Number of additional tasks

# Initialize the dataset and dataloaders
dataset = SMILESDataset("smiles_data1000_with_qed.csv", vocab_size=src_vocab_size, max_length=max_seq_length)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = MultiTaskTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks)
criterion_reconstruction = nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.token_to_id("[PAD]"))
criterion_tasks = nn.MSELoss()  # Assuming regression tasks for simplicity
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train(model, train_loader, criterion_reconstruction, criterion_tasks, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch in train_loader:
        src, task_targets = batch
        src = src.to(torch.long)
        task_targets = task_targets.to(torch.float)

        optimizer.zero_grad()
        tgt_input = src[:, :-1]  # Offset the target sequence by one
        tgt_output = src[:, 1:]  # Actual targets shifted by one

        output, task_outputs = model(src, tgt_input)

        # Compute reconstruction loss
        loss_reconstruction = criterion_reconstruction(output.view(-1, tgt_vocab_size), tgt_output.reshape(-1))

        # Compute task losses
        loss_tasks = sum([criterion_tasks(task_output, task_targets[:, i:i+1]) for i, task_output in enumerate(task_outputs)])

        # Combine losses
        loss = loss_reconstruction + loss_tasks
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}, Training Loss: {avg_loss}")

# Validation function
def validate(model, val_loader, criterion_reconstruction, criterion_tasks, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            src, task_targets = batch
            src = src.to(torch.long)
            task_targets = task_targets.to(torch.float)

            tgt_input = src[:, :-1]  # Offset the target sequence by one
            tgt_output = src[:, 1:]  # Actual targets shifted by one

            output, task_outputs = model(src, tgt_input)

            # Compute reconstruction loss
            loss_reconstruction = criterion_reconstruction(output.view(-1, tgt_vocab_size), tgt_output.reshape(-1))

            # Compute task losses
            loss_tasks = sum([criterion_tasks(task_output, task_targets[:, i:i+1]) for i, task_output in enumerate(task_outputs)])

            # Combine losses
            loss = loss_reconstruction + loss_tasks
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    print(f"Epoch {epoch}, Validation Loss: {avg_loss}")
    return avg_loss

# Main training loop with early stopping
best_val_loss = float('inf')
epochs_no_improve = 0

epoch = 0
while epochs_no_improve < patience:
    epoch += 1
    print(f"Epoch {epoch}")
    train(model, tqdm(train_loader, desc="Training"), criterion_reconstruction, criterion_tasks, optimizer, epoch)
    val_loss = validate(model, tqdm(val_loader, desc="Validating"), criterion_reconstruction, criterion_tasks, epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        epochs_no_improve += 1

print("Training stopped. No improvement in validation loss for {} epochs.".format(patience))
