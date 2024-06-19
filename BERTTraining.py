import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  # Import tqdm for progress bar
import time  # Import time for elapsed time calculation

# Create the dataset
dataset = ProteinDataset("data/uniprot1000.fasta", vocab_size=1000, max_length=512, mask_percentage=0.15)

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
num_epochs = 20
batch_size = 16

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

bert_model = BERT(vocab_size, d_model, max_len, n_heads, d_ff, n_layers)

# Define the optimizer and the loss function
optimizer = torch.optim.Adam(bert_model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index

# Function to evaluate the model
def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            b_input_ids = batch
            b_attention_masks = (b_input_ids != dataset.tokenizer.token_to_id("[PAD]")).long()  # Create attention mask
            b_labels = b_input_ids.clone()  # Use input_ids as labels for MLM task

            logits = model(b_input_ids, b_attention_masks)

            # Calculate loss only for masked tokens
            mask = (b_input_ids == dataset.tokenizer.token_to_id("[MASK]"))
            active_logits = logits.view(-1, vocab_size)
            active_labels = torch.where(mask.view(-1), b_labels.view(-1), torch.tensor(loss_fn.ignore_index).type_as(b_labels))
            loss = loss_fn(active_logits, active_labels)

            total_loss += loss.item()
    return total_loss / len(dataloader)

# Training loop
for epoch in range(num_epochs):  # Adjust number of epochs as needed
    start_time = time.time()
    bert_model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for batch in progress_bar:
        b_input_ids = batch
        b_attention_masks = (b_input_ids != 0).long()  # Create attention mask, assuming 0 is the [PAD] token ID

        # Forward pass
        logits = bert_model(b_input_ids, b_attention_masks)

        # Masking logic
        mask = (b_input_ids == dataset.tokenizer.token_to_id("[MASK]"))

        # Extract active logits and labels
        active_logits = logits[mask].view(-1, vocab_size)
        active_labels = b_input_ids[mask].view(-1)

        # Compute the loss
        loss = criterion(active_logits.view(-1, vocab_size), active_labels.view(-1))
        
        # Backward pass and optimization
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
    
    train_loss = total_loss / len(train_dataloader)
    # test_loss = evaluate(bert_model, test_dataloader, criterion)
    elapsed_time = time.time() - start_time

    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss:, Elapsed Time: {elapsed_time:.2f}s")
