import csv
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers
import torch
from torch.utils.data import Dataset

class SMILESDataset(Dataset):
    def __init__(self, file_path, vocab_size=1000, max_length=128):
        self.sequences, self.properties, self.tokenizer = self.process_data(file_path, vocab_size, max_length)
        self.max_length = max_length

    def process_data(self, file_path, vocab_size, max_length):
        def parse_smiles(file_path):
            smiles_list = []
            properties = []
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    smiles_list.append(row['SMILES'])
                    properties.append([float(row['logP']), float(row['tpsa']), float(row['h_donors']), float(row['h_acceptors']), float(row['solubility'])])  # Add your properties here
            return smiles_list, properties

        def train_bpe(sequences, vocab_size):
            tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.normalizer = normalizers.NFKC()
            trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[EOS]"], vocab_size=vocab_size)
            tokenizer.train_from_iterator(sequences, trainer)
            return tokenizer

        def add_special_tokens_pad(tokenizer, sequences, max_length):
            tokenized_sequences = []
            for seq in sequences:
                encoded = tokenizer.encode("[CLS] " + seq + " [EOS]")  # Add CLS at the beginning and EOS at the end
                ids = encoded.ids + [tokenizer.token_to_id("[PAD]")] * (max_length - len(encoded.ids))
                ids = ids[:max_length]  # Ensure length does not exceed max_length
                tokenized_sequences.append(ids)
            return tokenized_sequences

        sequences, properties = parse_smiles(file_path)
        tokenizer = train_bpe(sequences, vocab_size)
        tokenized_sequences = add_special_tokens_pad(tokenizer, sequences, max_length)

        return tokenized_sequences, properties, tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.properties[idx], dtype=torch.float)
