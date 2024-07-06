import csv
import random
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers
import torch
from torch.utils.data import Dataset
from rdkit import Chem

class SMILESDataset(Dataset):
    def __init__(self, file_path, vocab_size=1000, tokenizer=None, max_length=128):
        self.sequences, self.properties, self.tokenizer = self.process_data(file_path, vocab_size, max_length, tokenizer)
        self.max_length = max_length

    def process_data(self, file_path, vocab_size, max_length,  tokenizer=None):
        def parse_smiles(file_path):
            smiles_list = []
            properties = []
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # print(row)
                    smiles_list.append(row['SMILES'])
                    properties.append([float(value) for key, value in row.items() if key != 'SMILES'])  # Dynamically add all properties except 'SMILES'
                    # properties.append([float(row['logP']), float(row['tpsa']), float(row['h_donors']), float(row['h_acceptors']), float(row['solubility'])])  # Add your properties here
            return smiles_list, properties

        def train_bpe(sequences, vocab_size):
            tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.normalizer = normalizers.NFKC()
            trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[EOS]"], vocab_size=vocab_size)
            tokenizer.train_from_iterator(sequences, trainer)
            return tokenizer

        sequences, properties = parse_smiles(file_path)
        if tokenizer is None:
            tokenizer = train_bpe(sequences, vocab_size)

        return sequences, properties, tokenizer

    def randomize_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        randomized_smiles = Chem.MolToSmiles(mol, doRandom=True)
        return randomized_smiles

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        smiles = self.sequences[idx]
        randomized_smiles = self.randomize_smiles(smiles)  # Randomize the SMILES string
        encoded = self.tokenizer.encode("[CLS] " + randomized_smiles + " [EOS]")
        ids = encoded.ids + [self.tokenizer.token_to_id("[PAD]")] * (self.max_length - len(encoded.ids))
        ids = ids[:self.max_length]  # Ensure length does not exceed max_length
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.properties[idx], dtype=torch.float)
