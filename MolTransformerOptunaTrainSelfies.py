import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from tokenizers import Tokenizer
import yaml
from MolLoaderSelfies import SMILESDataset
from MolTransformerSelfies import MultiTaskTransformer
import optuna
import selfies as sf

def compute_normalization_factors(dataset):
    properties = []
    for _, prop in dataset:
        properties.append(prop.numpy())

    properties = np.array(properties)
    means = properties.mean(axis=0)
    stds = properties.std(axis=0)

    normalization_factors = {
        "fac1": {"mean": means[0], "std": stds[0]},
        "fac2": {"mean": means[1], "std": stds[1]},
        "fac3": {"mean": means[2], "std": stds[2]},
        "fac4": {"mean": means[3], "std": stds[3]},
        "fac5": {"mean": means[4], "std": stds[4]}
    }

    return normalization_factors

# class WarmupAnnealingLR:
#     def __init__(self, optimizer, warmup_steps, total_steps, learning_rate):
#         self.optimizer = optimizer
#         self.warmup_steps = warmup_steps
#         self.total_steps = total_steps
#         self.current_step = 0
#         self.learning_rate = learning_rate

#     def step(self):
#         self.current_step += 1
#         lr = self.compute_lr()
#         for param_group in self.optimizer.param_groups:
#             param_group['lr'] = lr

#     def compute_lr(self):
#         if self.current_step < self.warmup_steps:
#             return self.current_step / self.warmup_steps * self.learning_rate
#         else:
#             progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
#             return 0.5 * self.learning_rate * (1 + np.cos(np.pi * progress))

def get_lr(warmup_steps, total_steps, pretrain_steps, learning_rate, current_step, pretrain_learning_rate):
    if current_step < pretrain_steps:
        return pretrain_learning_rate
    elif current_step < pretrain_steps + warmup_steps:
        return (current_step - pretrain_steps) / warmup_steps * learning_rate
    else:
        progress = (current_step - warmup_steps - pretrain_steps) / (total_steps - warmup_steps - pretrain_steps)
        return 0.5 * learning_rate * (1 + np.cos(np.pi * progress))

# Function to normalize losses based on task-specific scales
def normalize_loss(loss, task_name,  normalization_factors):
    factor = normalization_factors[task_name]
    return loss / (factor['std']**2)

def train(model, train_loader, criterion_reconstruction, criterion_tasks, epoch, device, normalization_factors, num_tasks, tgt_vocab_size, warmup_steps, total_steps, max_lr, total_shared_params, total_head_params, pretrain_steps, pretrain_learning_rate, step):
    # print(f"step: {step}/{pretrain_steps}")

    model.train()
    total_loss = 0
    total_reconstruction_loss = 0
    total_task_losses = torch.zeros(num_tasks, device=device)
    # grads = [None] * (num_tasks + 1)  # Include reconstruction loss gradient norm
    # grads = torch.zeros([num_tasks + 1, total_shared_params], device=device)
    # grads_prev = None
    # grads_tasks = torch.zeros([total_head_params], device=device)
    # grads_tasks_prev = None

    optimizer_pretrain = optim.Adam(model.parameters(), lr=pretrain_learning_rate)

    mu_grads = torch.zeros([num_tasks + 1, total_shared_params], device=device)
    nu_grads = torch.zeros([num_tasks + 1, total_shared_params], device=device)
    mu_tasks = torch.zeros([total_head_params], device=device)
    nu_tasks = torch.zeros([total_head_params], device=device)

    epsilon = 1e-8
    beta = 0.9
    beta2 = 0.999
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
        step += 1
        learning_rate = get_lr(warmup_steps, total_steps, pretrain_steps, max_lr, step, pretrain_learning_rate)

        # run through model
        src, task_targets = batch
        src = src.to(torch.long).to(device)
        task_targets = task_targets.to(torch.float).to(device)
        tgt_input = src[:, :-1]  # Offset the target sequence by one
        tgt_output = src[:, 1:]  # Actual targets shifted by one
        output, task_outputs = model(src, tgt_input)

        # Compute reconstruction loss
        loss_reconstruction = criterion_reconstruction(output.view(-1, tgt_vocab_size), tgt_output.reshape(-1))

        # Compute task losses
        losses_tasks = [criterion_tasks(task_outputs[:, i], task_targets[:, i]) for i in range(num_tasks)]
        losses_tasks = torch.stack(losses_tasks)

        # grads_prev = grads.copy()
        # grads_prev = grads.clone()
        # grads_tasks_prev = grads_tasks.clone()
        # Combine reconstruction loss with task losses
        all_losses = torch.cat([loss_reconstruction.unsqueeze(0), losses_tasks])
        
        # normalize the loss
        normalized_losses = torch.zeros((all_losses.shape), device=device)
        for m, task_name in enumerate(normalization_factors):
            normalized_losses[m + 1] = all_losses[m + 1] / (normalization_factors[task_name]['std']**2)

        if step <= pretrain_steps:
            for param in model.parameters():
                param.grad = None
            
            loss_reconstruction.backward()
            
            list_grads = []
            for name, p in model.named_parameters():
                if p.grad is not None:
                    if not name.startswith("task_heads"):
                        list_grads.append(p.grad.view(-1))
            grads = torch.cat(list_grads)
            
            mu_grads[0] = beta * mu_grads[0] + (1 - beta) * grads
            nu_grads[0] = beta2 * nu_grads[0] + (1 - beta2) * (grads**2)

            mu_grads0_corr = mu_grads[0] / (1 - beta**step)
            nu_grads0_corr = nu_grads[0] / (1 - beta2 **step)
            mu_grads0_final = mu_grads0_corr / (torch.sqrt(nu_grads0_corr) + epsilon)
            with torch.no_grad():  # Disable gradient tracking
                idx = 0
                for name, p in model.named_parameters():
                    if not name.startswith("task_heads"):
                        numel = p.data.numel()
                        grad = mu_grads0_final[idx:idx + numel].view(p.shape)  # Reshape to original parameter shape
                        p.data -= learning_rate * grad  # Update the parameter
                        idx += numel

            # optimizer_pretrain.zero_grad()
            # loss_reconstruction.backward()
            # optimizer_pretrain.step()

            total_reconstruction_loss += loss_reconstruction.item()
            total_loss += torch.sum(normalized_losses).item() + loss_reconstruction.item()
            total_task_losses += normalized_losses[1:].detach()
            
            continue
        elif step == pretrain_steps + 1:
            print("PRETRAINING COMPLETE")

        all_losses = torch.log(all_losses + epsilon)
        # g_norms = [None] * (num_tasks + 1)
        # g_norms = torch.zeros(num_tasks + 1, device=device)

        # grads_tasks = [None] * (num_tasks * 2)
        # grads_tasks = [p.grad.view(-1) for name, p in model.named_parameters() if p.grad is not None and  name.startswith("task_heads")]
        # Compute and update g_t                    
        
        list_grads_tasks = []
        grads = torch.zeros([num_tasks + 1, total_shared_params], device=device)
        for i in range(num_tasks + 1):
            for param in model.parameters():
                param.grad = None

            all_losses[i].backward(retain_graph=True)
            list_grads = []
            for name, p in model.named_parameters():
                if p.grad is not None:
                    if not name.startswith("task_heads"):
                        list_grads.append(p.grad.view(-1))
                    elif i >= 0 and name.startswith(f"task_heads.{i - 1}"):
                        list_grads_tasks.append(p.grad.view(-1))

            grads[i] = torch.cat(list_grads)

        grads_tasks = torch.cat(list_grads_tasks)

        # if step != 1:
            # for i in range(num_tasks + 1):
                # grads[i] = beta * grads_prev[i] + (1 - beta) * grads[i]
            # grads = beta * grads_prev + (1 - beta) * grads

        # if step != 1:
            # grads_tasks = beta * grads_tasks_prev + (1 - beta) * grads_tasks

        mu_grads = beta * mu_grads + (1 - beta) * grads
        nu_grads = beta2 * nu_grads + (1 - beta2) * (grads**2)
        mu_grads_corr = torch.zeros(mu_grads.shape, device=device)
        nu_grads_corr = torch.zeros(nu_grads.shape, device=device)
        # mu_grads_corr = mu_grads / (1 - beta ** (step - pretrain_steps))
        # nu_grads_corr = nu_grads / (1 - beta2 ** (step - pretrain_steps))
        mu_grads_corr[0] = mu_grads[0] / (1 - beta ** step)
        nu_grads_corr[0] = nu_grads[0] / (1 - beta2 ** step)
        for i in range(num_tasks):
            mu_grads_corr[i + 1] = mu_grads[i + 1] / (1 - beta ** (step - pretrain_steps))
            nu_grads_corr[i + 1] = nu_grads[i + 1] / (1 - beta2 ** (step - pretrain_steps))

        grads_final = mu_grads_corr / (torch.sqrt(nu_grads_corr) + epsilon)

        mu_tasks = beta * mu_tasks + (1 - beta) * grads_tasks
        nu_tasks = beta2 * nu_tasks + (1 - beta2) * (grads_tasks**2)
        mu_tasks_corr = mu_tasks / (1 - beta ** (step - pretrain_steps))
        nu_tasks_corr = nu_tasks / (1 - beta2 ** (step - pretrain_steps))

        # g_norms = torch.linalg.vector_norm(mu_grads, dim=1, keepdim=True)
        g_norms = torch.linalg.vector_norm(grads_final, dim=1, keepdim=True)
        # Max-norm strategy for adjusting alpha_k
        alpha_k = torch.max(g_norms)
        # weights = []
        # for i in range(num_tasks + 1):
        #     weights.append(1 / (g_norms[i] + epsilon))
        weights = alpha_k / (g_norms + epsilon)
        
        # grads_weighted = torch.zeros([num_tasks + 1, (grads[0].shape)[0]], device=device)
        # for i in range(num_tasks + 1):
        #     grads_weighted[i] = weights[i] * grads[i]

        total_grads = torch.sum(weights * grads_final, dim = 0)
        # total_grads = alpha_k * torch.sum(grads_weighted, dim=0)

        with torch.no_grad():  # Disable gradient tracking
            idx = 0
            idx2 = 0
            for name, p in model.named_parameters():
                if not name.startswith("task_heads"):
                    numel = p.data.numel()
                    grad = total_grads[idx:idx + numel].view(p.shape)  # Reshape to original parameter shape
                    p.data -= learning_rate * grad  # Update the parameter
                    idx += numel
                else:
                    numel = p.data.numel()
                    grad = (mu_tasks_corr[idx2:idx2 + numel] / (torch.sqrt(nu_tasks_corr[idx2:idx2+numel]) + epsilon)).view(p.shape)
                    p.data -= learning_rate * grad
                    idx2 += numel

        total_reconstruction_loss += loss_reconstruction.item()
        total_loss += torch.sum(normalized_losses).item() + loss_reconstruction.item()
        total_task_losses += normalized_losses[1:].detach()

    avg_loss = total_loss / len(train_loader)
    avg_reconstruction_loss = total_reconstruction_loss / len(train_loader)
    avg_task_losses = total_task_losses / len(train_loader)

    print(f"Epoch {epoch}, Training Loss: {avg_loss}")
    print(f"Epoch {epoch}, Training Reconstruction Loss: {avg_reconstruction_loss}")
    for i, task_name in enumerate(normalization_factors):
        print(f"Epoch {epoch}, Training Task {i+1} Loss: {avg_task_losses[i]} (Normalized)")
    return step

# Validation function
def validate(model, val_loader, criterion_reconstruction, criterion_tasks, epoch, device, normalization_factors, num_tasks, tgt_vocab_size):
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

            output, task_outputs = model(src, tgt_input)

            # Compute reconstruction loss
            loss_reconstruction = criterion_reconstruction(output.view(-1, tgt_vocab_size), tgt_output.reshape(-1))
            # Compute task losses
            losses_tasks = [criterion_tasks(task_outputs[:, i], task_targets[:, i]) for i in range(num_tasks)]
            losses_tasks = torch.stack(losses_tasks)
            
            # Combine losses
            all_losses = torch.cat([loss_reconstruction.unsqueeze(0), losses_tasks])
            for m, task_name in enumerate(normalization_factors):
                all_losses[m + 1] = all_losses[m + 1] / (normalization_factors[task_name]['std']**2)
            
            total_loss += torch.sum(all_losses).item()
            total_reconstruction_loss += loss_reconstruction.item()
            total_task_losses += all_losses[1:].detach()

    avg_loss = total_loss / len(val_loader)
    avg_reconstruction_loss = total_reconstruction_loss / len(val_loader)
    avg_task_losses = total_task_losses / len(val_loader)

    print(f"Epoch {epoch}, Validation Loss: {avg_loss}")
    print(f"Epoch {epoch}, Validation Reconstruction Loss: {avg_reconstruction_loss}")
    for i, task_name in enumerate(normalization_factors):
        print(f"Epoch {epoch}, Validation Task {i+1} Loss: {avg_task_losses[i]} (Normalized)")
    return avg_loss

def train_and_validate(d_model, num_heads, num_layers, d_ff, dropout, learning_rate, batch_size, device, warmup_epochs, total_epochs, patience, pretrain_epochs, pretrain_learning_rate, trial=None):
    # unchanging hyperparams
    with open("hyperparams.yaml", "r") as file:
        config = yaml.safe_load(file)

    src_vocab_size = config["mol_model"]["src_vocab_size"]
    tgt_vocab_size = config["mol_model"]["tgt_vocab_size"]
    max_seq_length = config["mol_model"]["max_seq_length"]
    num_tasks = config["mol_model"]["num_tasks"]
    # tok_file = config["mol_model"]["tokenizer_file"]

    # Initialize the dataset and dataloaders
    file_path = "data/smiles_10000_selected_features_cleaned.csv"
    # smiles_tokenizer = Tokenizer.from_file(tok_file)
    dataset = SMILESDataset(file_path, vocab_size=79, max_length=128, tokenizer_path="models/selfies_tok.json")
    normalization_factors = compute_normalization_factors(dataset)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    warmup_steps = int(warmup_epochs * train_size / batch_size)
    total_steps = int(total_epochs * train_size / batch_size)
    pretrain_steps = int(pretrain_epochs * train_size / batch_size)

    # Initialize the model, loss function, and optimizer
    model = MultiTaskTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks).to(device)
    criterion_reconstruction = nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.token_to_id["[PAD]"])
    criterion_tasks = nn.MSELoss()
    
    # model.load_state_dict(torch.load('models/selfies_transformer.pt', map_location=device))
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = WarmupAnnealingLR(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, learning_rate=learning_rate)

    total_shared_params = 0
    total_head_params = 0
    for name, param in model.named_parameters():
        if name.startswith("task_heads"):
            total_head_params += torch.numel(param)
        else:
            total_shared_params += torch.numel(param)

    # Main training loop with early stopping and learning rate scheduler
    best_val_loss = float('inf')
    epochs_no_improve = 0

    epoch = 0
    step = 0
    while epochs_no_improve < patience:
        epoch += 1

        if(epoch >= total_epochs):
            break

        print(f"Epoch {epoch}")
        step = train(model, train_loader, criterion_reconstruction, criterion_tasks, epoch, device, normalization_factors, num_tasks, tgt_vocab_size, warmup_steps, total_steps, learning_rate, total_shared_params, total_head_params, pretrain_steps, pretrain_learning_rate, step)
        val_loss = validate(model, val_loader, criterion_reconstruction, criterion_tasks, epoch, device, normalization_factors, num_tasks, tgt_vocab_size)

        # if not trial is None and epoch > pretrain_epochs + 10 and epoch >= 20:
        #     trial.report(val_loss, epoch)
        #     if trial.should_prune():
        #         raise optuna.TrialPruned()
        
        if epoch > pretrain_epochs:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save the best model
                torch.save(model.state_dict(), 'models/selfies_transformer_final.pt')
            else:
                epochs_no_improve += 1

    print("Training stopped. No improvement in validation loss for {} epochs.".format(patience))
    return best_val_loss


def main():
    with open("hyperparams.yaml", "r") as file:
        config = yaml.safe_load(file)
        
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

    train_and_validate(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        patience=patience,
        pretrain_epochs=pretrain_epochs,
        pretrain_learning_rate=pretrain_learning_rate
    )

main()