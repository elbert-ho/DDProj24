import csv
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers

def parse_smiles(file_path):
    smiles_list = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles_list.append(row['SMILES'])
    return smiles_list

def train_bpe(sequences, vocab_size):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.normalizer = normalizers.NFKC()
    trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[EOS]"], vocab_size=vocab_size)
    tokenizer.train_from_iterator(sequences, trainer)
    return tokenizer

file_path = '../data/smiles_10000_selected_features.csv'
vocab_size = 1000
sequences = parse_smiles(file_path)
tokenizer = train_bpe(sequences, vocab_size)
save_path = '../models/smiles_tokenizer_10K_full.json'
tokenizer.save(save_path)