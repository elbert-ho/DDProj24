from Bio import SeqIO
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import numpy as np
import torch
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, file_path, vocab_size=1000, max_length=512, mask_percentage=0.15):
        tup = self.process_data(file_path, vocab_size, max_length, mask_percentage)
        self.data = tup[0]
        self.tokenizer = tup[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)
    
    def process_data(file_path, vocab_size, max_length, mask_percentage):
        def parse_fasta(file_path):
            sequences = []
            for record in SeqIO.parse(file_path, "fasta"):
                sequences.append(str(record.seq))
            return sequences

        def train_bpe(sequences, vocab_size):
            tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
            trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[EOS]", "[MASK]", "[UNK]"], vocab_size=vocab_size)
            tokenizer.train_from_iterator(sequences, trainer)
            return tokenizer

        def add_special_tokens_pad(tokenizer, sequences, max_length):
            tokenized_sequences = []

            for seq in sequences:
                encoded = tokenizer.encode("[CLS] " + seq + " [EOS]")
                ids = encoded.ids + [tokenizer.token_to_id("[PAD]")] * (max_length - len(encoded.ids))
                ids = ids[:max_length]  # Ensure length does not exceed max_length
                tokenized_sequences.append(ids)

            return tokenized_sequences

        def apply_mask(tokenized_sequences, tokenizer, mask_percentage):
            mask_token_id = tokenizer.token_to_id("[MASK]")
            masked_sequences = []

            for ids in tokenized_sequences:
                num_to_mask = int((len(ids) - 2) * mask_percentage)  # -2 to avoid [CLS] and [EOS]
                mask_indices = np.random.choice(len(ids) - 2, num_to_mask, replace=False) + 1  # Avoid [CLS] and [EOS]

                masked_ids = ids[:]
                for idx in mask_indices:
                    if masked_ids[idx] != tokenizer.token_to_id("[PAD]"):  # Avoid masking [PAD]
                        masked_ids[idx] = mask_token_id

                masked_sequences.append(masked_ids)

            return masked_sequences

        sequences = parse_fasta(file_path)
        tokenizer = train_bpe(sequences, vocab_size)
        tokenized_sequences = add_special_tokens_pad(tokenizer, sequences, max_length)
        masked_sequences = apply_mask(tokenized_sequences, tokenizer, mask_percentage)

        return masked_sequences, tokenizer