# Define the BERT model architecture
class BERT(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, n_heads, d_ff, dropout_rate, n_layers):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size, d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_model // n_heads, d_model // n_heads, n_heads, d_ff, dropout_rate) for _ in range(n_layers)])
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, input_ids, attention_masks):
        output = self.embedding(input_ids)
        for layer in self.layers:
            output = layer(output, attention_masks)
        return self.classifier(output[:, 0])  # Use the [CLS] token to classify sentiment

# Create a small sentiment analysis dataset
sentences = [
    "I love this product!", "This is an amazing movie!", "I feel great today!", "What a beautiful day!",
    "I am so excited about this!", "Absolutely perfect! Highly recommend!", "I enjoyed every moment.",
    "This book is a masterpiece!", "The experience was wonderful.", "She is incredibly happy.",
    "I hate this product!", "This is a terrible movie!", "I feel awful today!", "What a horrible day!",
    "I am so disappointed about this!", "Absolutely awful! Do not recommend!", "I regret every moment.",
    "This book is terrible!", "The experience was dreadful.", "She is incredibly sad."
]
labels = [1] * 10 + [0] * 10  # 1 for positive, 0 for negative

# Tokenization and encoding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
input_ids = encoded_inputs['input_ids']
attention_masks = encoded_inputs['attention_mask']

# Convert labels to tensor and create a dataset
labels = torch.tensor(labels)
dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize the BERT model
model = BERT(vocab_size=30522, d_model=768, max_len=512, n_heads=8, d_ff=2048, dropout_rate=0.1, n_layers=12)

# Define the optimizer and the loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(20):
    total_loss = 0
    for batch in dataloader:
        b_input_ids, b_attention_masks, b_labels = batch
        model.zero_grad()
        logits = model(b_input_ids, b_attention_masks)
        loss = loss_fn(logits, b_labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Test inference
model.eval()
with torch.no_grad():
    new_sentence = "The weather is really quite nice outside."
    new_encoded = tokenizer(new_sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    new_input_ids = new_encoded['input_ids']
    new_attention_masks = new_encoded['attention_mask']
    logits = model(new_input_ids, new_attention_masks)
    prediction = torch.argmax(logits, dim=1)
    print("Predicted sentiment (1 positive, 0 negative):", prediction.item())


