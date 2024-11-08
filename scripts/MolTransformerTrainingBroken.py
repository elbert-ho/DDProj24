import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from MolLoader import SMILESDataset
from MolTransformer import MultiTaskTransformer
from tokenizers import Tokenizer
from torch.cuda.amp import GradScaler, autocast

def compute_normalization_factors(dataset):
    properties = []
    for _, prop in dataset:
        properties.append(prop.numpy())

    properties = np.array(properties)
    means = properties.mean(axis=0)
    stds = properties.std(axis=0)

    normalization_factors = {
        "logP": {"mean": means[0], "std": stds[0]},
        "tpsa": {"mean": means[1], "std": stds[1]},
        "h_donors": {"mean": means[2], "std": stds[2]},
        "h_acceptors": {"mean": means[3], "std": stds[3]},
        "solubility": {"mean": means[4], "std": stds[4]}
    }

    return normalization_factors

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
patience = 5  # Early stopping patience
num_tasks = 5  # Number of additional tasks

# Initialize the dataset and dataloaders
file_path = "data/smiles_10000_with_props.csv"
smiles_tokenizer = Tokenizer.from_file('models/smiles_tokenizer.json')
dataset = SMILESDataset(file_path, vocab_size=1000, max_length=128, tokenizer=smiles_tokenizer)
normalization_factors = compute_normalization_factors(dataset)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Initialize the model, loss function, and optimizer
model = MultiTaskTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks).to(device)
criterion_reconstruction = nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.token_to_id("[PAD]"))
criterion_tasks = nn.MSELoss()  # Assuming regression tasks for simplicity
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize average losses
average_losses = torch.ones(num_tasks + 1, device=device)  # Including reconstruction loss

# Learning rate scheduler with warmup and annealing
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class WarmupAnnealingLR:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def compute_lr(self):
        if self.current_step < self.warmup_steps:
            return self.current_step / self.warmup_steps * learning_rate
        else:
            return learning_rate * (1 - (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps))

scheduler = WarmupAnnealingLR(optimizer, warmup_steps=1000, total_steps=20000)

# Function to normalize losses based on task-specific scales
def normalize_loss(loss, task_name):
    factor = normalization_factors[task_name]
    return loss / (factor['mean']**2)

g_t = torch.zeros(num_tasks + 1, device=device)  # Include reconstruction loss gradient norm
epsilon = 1e-8

# Initialize GradScaler
scaler = GradScaler()

# Training function with GradNorm
def train(model, train_loader, criterion_reconstruction, criterion_tasks, optimizer, scheduler, epoch, device):
    model.train()
    total_loss = 0
    total_reconstruction_loss = 0
    total_task_losses = torch.zeros(num_tasks, device=device)

    beta = 0.9

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
        src, task_targets = batch
        src = src.to(torch.long).to(device, non_blocking=True)
        task_targets = task_targets.to(torch.float).to(device, non_blocking=True)

        tgt_input = src[:, :-1]  # Offset the target sequence by one
        tgt_output = src[:, 1:]  # Actual targets shifted by one

        optimizer.zero_grad()

        with autocast():
            output, task_outputs = model(src, tgt_input)

            # Compute reconstruction loss
            loss_reconstruction = criterion_reconstruction(output.view(-1, tgt_vocab_size), tgt_output.reshape(-1))

            # Compute task losses
            losses_tasks = [criterion_tasks(task_outputs[:, i], task_targets[:, i]) for i in range(num_tasks)]
            losses_tasks = torch.stack(losses_tasks)

            # Combine reconstruction loss with task losses
            all_losses = torch.cat([loss_reconstruction.unsqueeze(0), losses_tasks])
            
            global g_t
            # Compute and update g_t
            g_t_prev = g_t.clone()
            for i in range(num_tasks + 1):
                all_losses[i].backward(retain_graph=True)
                g_t[i] = torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]))

            g_t = beta * g_t_prev + (1 - beta) * g_t

            # Max-norm strategy for adjusting alpha_k
            alpha_k = g_t.max()

            # Compute the weighted sum of losses using logarithm transformation
            weights = alpha_k / (g_t + epsilon)
            weighted_losses = torch.sum(weights * torch.log1p(all_losses))

        # Scale the loss and call backward()
        scaler.scale(weighted_losses).backward()
        # Unscale the gradients and perform the optimization step
        scaler.step(optimizer)


        scaler.update()
        scheduler.step()

        total_loss += weighted_losses.item()
        total_reconstruction_loss += loss_reconstruction.item()
        total_task_losses += losses_tasks.detach()

    avg_loss = total_loss / len(train_loader)
    avg_reconstruction_loss = total_reconstruction_loss / len(train_loader)
    avg_task_losses = total_task_losses / len(train_loader)

    # Normalize task losses for printing
    normalized_task_losses = [normalize_loss(avg_task_losses[i], task_name) for i, task_name in enumerate(normalization_factors)]

    print(f"Epoch {epoch}, Training Loss: {avg_loss}")
    print(f"Epoch {epoch}, Training Reconstruction Loss: {avg_reconstruction_loss}")
    for i, task_name in enumerate(normalization_factors):
        print(f"Epoch {epoch}, Training Task {i+1} Loss: {normalized_task_losses[i]} (Normalized)")

# Validation function
def validate(model, val_loader, criterion_reconstruction, criterion_tasks, epoch, device):
    model.eval()
    total_loss = 0
    total_reconstruction_loss = 0
    total_task_losses = torch.zeros(num_tasks, device=device)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch}"):
            src, task_targets = batch
            src = src.to(torch.long).to(device)
            task_targets = task_targets.to(torch.float).to(device)

            tgt_input = src[:, :-1]  # Offset the target sequence by one
            tgt_output = src[:, 1:]  # Actual targets shifted by one

            with autocast():
                output, task_outputs = model(src, tgt_input)

                # Compute reconstruction loss
                loss_reconstruction = criterion_reconstruction(output.view(-1, tgt_vocab_size), tgt_output.reshape(-1))

                # Compute task losses
                losses_tasks = [criterion_tasks(task_outputs[:, i], task_targets[:, i]) for i in range(num_tasks)]
                losses_tasks = torch.stack(losses_tasks)
                
                all_losses = torch.cat([loss_reconstruction.unsqueeze(0), losses_tasks])
                
                # Combine losses
                global g_t
                alpha_k = g_t.max()
                weights = alpha_k / (g_t + epsilon)
                weighted_losses = torch.sum(weights * torch.log1p(all_losses))
                total_loss += weighted_losses.item()

            total_reconstruction_loss += loss_reconstruction.item()
            total_task_losses += losses_tasks

    avg_loss = total_loss / len(val_loader)
    avg_reconstruction_loss = total_reconstruction_loss / len(val_loader)
    avg_task_losses = total_task_losses / len(val_loader)

    # Normalize task losses for printing
    normalized_task_losses = [normalize_loss(avg_task_losses[i], task_name) for i, task_name in enumerate(normalization_factors)]

    print(f"Epoch {epoch}, Validation Loss: {avg_loss}")
    print(f"Epoch {epoch}, Validation Reconstruction Loss: {avg_reconstruction_loss}")
    for i, task_name in enumerate(normalization_factors):
        print(f"Epoch {epoch}, Validation Task {i+1} Loss: {normalized_task_losses[i]} (Normalized)")
    return avg_loss

if __name__ == "__main__":
    # Main training loop with early stopping and learning rate scheduler
    best_val_loss = float('inf')
    epochs_no_improve = 0

    epoch = 0
    while epochs_no_improve < patience:
        epoch += 1
        print(f"Epoch {epoch}")
        train(model, train_loader, criterion_reconstruction, criterion_tasks, optimizer, scheduler, epoch, device)
        val_loss = validate(model, val_loader, criterion_reconstruction, criterion_tasks, epoch, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            if epoch > 30:
                torch.save(model.state_dict(), 'models/best_model.pt')
        else:
            epochs_no_improve += 1

    print("Training stopped. No improvement in validation loss for {} epochs.".format(patience))
