import csv
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from rdkit import Chem
from SelfiesTok import SelfiesTok
import selfies as sf

class SMILESDataset(Dataset):
    def __init__(self, file_path, vocab_size=256, tokenizer_path=None, max_length=128):
        self.vocab_size = vocab_size
        self.sequences, self.tokenizer = self.process_data(file_path, tokenizer_path)
        self.max_length = max_length

    def process_data(self, file_path, tokenizer_path=None):
        def parse_smiles(file_path):
            smiles_list = []
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    smiles_list.append(row['SMILES'])
            return smiles_list

        sequences = parse_smiles(file_path)
        
        tokenizer = None
        if tokenizer_path is not None:
            tokenizer = SelfiesTok.load(tokenizer_path)
        else:
            selfies_dataset = list(map(sf.encoder, sequences))
            tokenizer = SelfiesTok(selfies_dataset, target_vocab_size=self.vocab_size)
            tokenizer.save("models/selfies_tok_bpe.json")

        return sequences, tokenizer

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
        while True:
            try:
                randomized_smiles = self.randomize_smiles(smiles)  # Randomize the SMILES string
                selfies_toks = sf.split_selfies(sf.encoder(randomized_smiles))
                break
            except:
                continue

        encoded = self.tokenizer.encode(["[CLS]"] + list(selfies_toks) + ["[EOS]"])
        # Pad the encoded tokens to ensure max_length
        ids = encoded + [self.tokenizer.token_to_id["[PAD]"]] * (self.max_length - len(encoded))
        ids = ids[:self.max_length]  # Ensure length does not exceed max_length

        # Convert ids to one-hot encoding
        one_hot = self.ids_to_one_hot(ids)

        return torch.tensor(ids, dtype=torch.long), torch.tensor(one_hot, dtype=torch.float32)

    def ids_to_one_hot(self, ids):
        """ Convert token ids to one-hot encoding """
        one_hot = np.zeros((self.max_length, self.vocab_size), dtype=np.float32)
        for i, token_id in enumerate(ids):
            if token_id < self.vocab_size:  # Ensure the token ID is within bounds
                one_hot[i, token_id] = 1.0
        return one_hot
