from Bio import SeqIO
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

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

file_path = '../data/uniprot1000.fasta'
vocab_size = 1000
sequences = parse_fasta(file_path)
tokenizer = train_bpe(sequences, vocab_size)
save_path = '../models/protein_tokenizer.json'
tokenizer.save(save_path)