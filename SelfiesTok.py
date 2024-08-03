import json
from selfies import get_alphabet_from_selfies, split_selfies

class SelfiesTok:
    def __init__(self, selfies_list=None, vocab=None):
        if selfies_list:
            alphabet = get_alphabet_from_selfies(selfies_list)
            self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[EOS]"] + sorted(alphabet)
            self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
            self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        elif vocab:
            self.vocab = vocab
            self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
            self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        else:
            raise ValueError("Either selfies_list or vocab must be provided.")

    def encode(self, selfies_toks):
        # tokens = split_selfies(selfies_string)
        token_ids = [self.token_to_id.get(token, self.token_to_id["[UNK]"]) for token in selfies_toks]
        return token_ids

    def decode(self, token_ids, skip_special_tokens):
        tokens = [self.id_to_token.get(id, "[UNK]") for id in token_ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in ["[PAD]", "[UNK]", "[CLS]", "[EOS]"]]
        selfies_string = ''.join(tokens)
        return selfies_string

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            vocab = json.load(f)
        return cls(vocab=vocab)