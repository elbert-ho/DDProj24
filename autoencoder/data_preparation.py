import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler

import numpy as np
import pandas as pd
from rdkit import Chem
import selfies as sf
from SelfiesTok import SelfiesTok
import csv

class DatasetBuilding(Dataset):
    def __init__(self, filepath, max_seq_len, data_type='pure_smiles', tokenizer=None, vocab_size=79):
        self.filepath = filepath
        self.data, self.tokenizer = self.process_data(filepath, vocab_size, max_seq_len, tokenizer)
        self.max_seq_len = max_seq_len
        self.data_type = data_type
        self.len = len(self.data)
        self.vocab_size = vocab_size

        # print(self.enumerate_smiles(self.data))

        # self.enum_smiles = self.enumerate_smiles(self.data)

    def process_data(self, file_path, vocab_size, max_length, tokenizer_path=None):
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
            tokenizer = SelfiesTok(selfies_dataset)
            tokenizer.save("models/selfies_tok.json")

        return sequences, tokenizer

    def __getitem__(self, index):
        smiles = self.data[index]
        # Randomize the SMILES and convert to SELFIES

        while(True):
            try:
                randomized_smiles = self.randomize_smiles(smiles)
                selfies = sf.encoder(randomized_smiles)
                break
            except:
                continue

        
        # Tokenize the SELFIES
        selfies_tokens = sf.split_selfies(selfies)
        encoded_tokens = self.tokenizer.encode(["[CLS]"] + list(selfies_tokens) + ["[EOS]"])
        
        # Pad the tokens to max_seq_len
        inputs_padd = Variable(torch.zeros((1, self.max_seq_len))).long()
        ids = encoded_tokens + [self.tokenizer.token_to_id["[PAD]"]] * (self.max_seq_len - len(encoded_tokens))
        inputs_padd[0, :len(ids)] = torch.LongTensor(ids[:self.max_seq_len])
        
        targets = inputs_padd[0, 1:]
        inputs = inputs_padd[0, :-1]
        seq_len = len(ids) - 1
        
        # Create the sample based on the data_type
        if self.data_type == 'physchem_biology':
            effect = self.data['Effect'][index]
            physchem = self.data['physchem'][index]
            sample = {'input': inputs, 'target': targets, 'length': seq_len, 'Effect': effect, 'physchem': torch.FloatTensor(physchem)}
        elif self.data_type == 'physchem':
            physchem = self.data['physchem'][index]
            sample = {'input': inputs, 'target': targets, 'length': seq_len, 'physchem': torch.FloatTensor(physchem)}
        elif self.data_type == 'biology':
            effect = self.data['Effect'][index]
            sample = {'input': inputs, 'target': targets, 'length': seq_len, 'Effect': effect}
        elif self.data_type == 'pure_smiles':
            sample = {'input': inputs, 'target': targets, 'length': seq_len}
        elif self.data_type == 'smiles_properties':
            prop = self.data['prop'][index]
            sample = {'input': inputs, 'target': targets, 'length': seq_len, 'prop': prop}
        else:
            raise ValueError('Requested class does not exist!')
        
        return sample

    def __len__(self):
        return self.len

    def enumerate_smiles(self, smiles_list):
        """Generates enumerated SMILES for each molecule to create diverse representations."""
        enumerated_smiles = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                enumerated_smiles.extend([Chem.MolToSmiles(mol, doRandom=True) for _ in range(5)])  # 5 variations per SMILES
            else:
                enumerated_smiles.append(smi)  # Fallback if parsing fails
        return enumerated_smiles

    def randomize_smiles(self, smiles):
        """Randomizes the SMILES string using RDKit."""
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, doRandom=True)
        return smiles  # Return the original if randomization fails

def weightSampler(data):
    """Creates a weighted sampler for imbalanced datasets."""
    class_sample_count = np.array([len(np.where(data['Effect'] == t)[0]) for t in np.unique(data['Effect'])])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in data['Effect']])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    return sampler
