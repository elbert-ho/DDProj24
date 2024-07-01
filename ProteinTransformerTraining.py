import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  # Import tqdm for progress bar
import time  # Import time for elapsed time calculation
import torch_optimizer as optim
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
from ProteinLoader import *
from ProteinTransformer import *

class CombinedScheduler:
    def __init__(self, warmup_scheduler, annealing_scheduler, warmup_steps):
        self.warmup_scheduler = warmup_scheduler
        self.annealing_scheduler = annealing_scheduler
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        if self.current_step < self.warmup_steps:
            self.warmup_scheduler.step()
        else:
            self.annealing_scheduler.step()
        self.current_step += 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the dataset
dataset = ProteinDataset("uniprot1000.fasta", vocab_size=1000, max_length=512, mask_percentage=0.15)

# Split dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Initialize the BERT model, smaller versions
vocab_size = 1000  # Adjusted for this specific dataset
d_model = 768
max_len = 512
n_heads = 8
d_ff = 512
n_layers = 5
batch_size = 16
# Early stopping parameters
patience = 3  # Number of epochs to wait for improvement
best_loss = float('inf')
epochs_no_improve = 0
num_epochs = 15  # Adjust number of epochs as needed

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

bert_model = BERT(vocab_size, d_model, max_len, n_heads, d_ff, n_layers).to(device)

# Define the optimizer and the loss function
optimizer = optim.Lamb(bert_model.parameters(), lr=2e-5)

num_training_steps = num_epochs * len(train_dataloader)
num_warmup_steps = int(0.1 * num_training_steps)

# Create a combined scheduler with warmup and cosine annealing
scheduler_warmup = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
scheduler_annealing = CosineAnnealingLR(optimizer, T_max=(num_training_steps - num_warmup_steps))
scheduler = CombinedScheduler(scheduler_warmup, scheduler_annealing, num_warmup_steps)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index

# Function to evaluate the model
def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            b_input_ids = batch.to(device)
            b_attention_masks = (b_input_ids != dataset.tokenizer.token_to_id("[PAD]")).long().to(device)  # Create attention mask
            b_labels = b_input_ids.clone().to(device)  # Use input_ids as labels for MLM task

            logits = model(b_input_ids, b_attention_masks)[0]

            # Calculate loss only for masked tokens
            mask = (b_input_ids == dataset.tokenizer.token_to_id("[MASK]")).to(device)
            active_logits = logits.view(-1, vocab_size)
            active_labels = torch.where(mask.view(-1), b_labels.view(-1), torch.tensor(loss_fn.ignore_index).type_as(b_labels).to(device))
            loss = loss_fn(active_logits, active_labels)

            total_loss += loss.item()
    return total_loss / len(dataloader)

for epoch in range(num_epochs):
    start_time = time.time()
    bert_model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for batch in progress_bar:
        b_input_ids = batch.to(device)
        b_attention_masks = (b_input_ids != 0).long().to(device)  # Create attention mask, assuming 0 is the [PAD] token ID

        # Forward pass
        logits = bert_model(b_input_ids, b_attention_masks)[0]

        # Masking logic
        mask = (b_input_ids == dataset.tokenizer.token_to_id("[MASK]")).to(device)

        # Extract active logits and labels
        active_logits = logits[mask].view(-1, vocab_size)
        active_labels = b_input_ids[mask].view(-1)

        # Compute the loss
        loss = criterion(active_logits, active_labels)

        # Backward pass and optimization
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

    train_loss = total_loss / len(train_dataloader)

    # Validation step (assuming you have a function `evaluate` to calculate validation loss)
    val_loss = evaluate(bert_model, test_dataloader, criterion)

    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Elapsed Time: {elapsed_time:.2f}s")

    # Early stopping logic
    if val_loss < best_loss:
        best_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

torch.save(bert_model.state_dict(), "model.pth")
