import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from selfies import get_alphabet_from_selfies, split_selfies

class SelfiesTok:
    def __init__(self, selfies_list=None, vocab=None, merges=None, target_vocab_size=None):
        if selfies_list and target_vocab_size:
            self.tokenizer = self.train_bpe(selfies_list, target_vocab_size)
        elif vocab and merges:
            self.tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges))
        else:
            raise ValueError("Either selfies_list and target_vocab_size or vocab and merges must be provided.")
        
    def train_bpe(self, selfies_list, target_vocab_size):
        # Prepare the data for training
        selfies_strings = [" ".join(split_selfies(selfies)) for selfies in selfies_list]

        # Create a tokenizer object with BPE
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Create a BPE trainer and train it on the SELFIES data
        trainer = trainers.BpeTrainer(vocab_size=target_vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[EOS]"])
        tokenizer.train_from_iterator(selfies_strings, trainer)

        return tokenizer

    def encode(self, selfies_toks):
        """ Expecting selfies_toks to already be a list of tokens """
        encoded = self.tokenizer.encode(" ".join(selfies_toks))
        return encoded.ids

    def decode(self, token_ids, skip_special_tokens=False):
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return decoded.replace(" ", "")

    def save(self, filepath):
        self.tokenizer.save(filepath)

    @classmethod
    def load(cls, filepath):
        tokenizer = Tokenizer.from_file(filepath)
        return cls(vocab=tokenizer.get_vocab(), merges=tokenizer.model.get_merges())
