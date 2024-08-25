import csv
import random
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from SelfiesTok import SelfiesTok
import selfies as sf

class SMILESDataset(Dataset):
    def __init__(self, file_path, vocab_size=79, tokenizer_path=None, max_length=128):
        self.sequences, self.properties, self.tokenizer = self.process_data(file_path, vocab_size, max_length, tokenizer_path)
        self.max_length = max_length

    def process_data(self, file_path, vocab_size, max_length,  tokenizer_path=None):
        def parse_smiles(file_path):
            smiles_list = []
            properties = []
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # print(row)
                    smiles_list.append(row['SMILES'])
                    properties.append([float(value) for key, value in row.items() if key != 'SMILES'])  # Dynamically add all properties except 'SMILES'
            return smiles_list, properties

        sequences, properties = parse_smiles(file_path)
        
        tokenizer = None
        if tokenizer_path is not None:
            tokenizer = SelfiesTok.load(tokenizer_path)
        else:
            selfies_dataset = list(map(sf.encoder, sequences))
            # print(selfies_dataset[:10])
            tokenizer = SelfiesTok(selfies_dataset)
            tokenizer.save("models/selfies_tok.json")

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
        selfies_toks = sf.split_selfies(sf.encoder(randomized_smiles))
        encoded = self.tokenizer.encode(["[CLS]"] + list(selfies_toks) + ["[EOS]"])
        # print(self.tokenizer.token_to_id)
        # print(selfies_toks)
        # print(encoded)
        ids = encoded + [self.tokenizer.token_to_id["[PAD]"]] * (self.max_length - len(encoded))
        ids = ids[:self.max_length]  # Ensure length does not exceed max_length
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.properties[idx], dtype=torch.float)
