import csv
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import torch
from torch.utils.data import Dataset
import selfies as sf
from rdkit import Chem
import json

class SMILESDataset(Dataset):
    def __init__(self, file_path, vocab_size=256, max_length=128, tokenizer_path=None, props=True, unicode_path=None):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.props = props
        self.tokenizer_path = tokenizer_path
        if unicode_path is None:
            self.unicode_mapping = {}
            self.reverse_unicode_mapping = {}
        else:
            with open(unicode_path, "r") as f:
                self.unicode_mapping = json.load(f)
                self.reverse_unicode_mapping = {v: k for k, v in self.unicode_mapping.items()}
        self.sequences, self.properties, self.tokenizer = self.process_data(file_path, vocab_size, tokenizer_path)

    def create_selfies_alphabet(self, smiles_list, num_randomizations=5):
        """Create an alphabet from SELFIES tokens and map to unique Unicode characters."""
        alphabet = set()
        # Randomize each SMILES several times and convert to SELFIES
        for smiles in smiles_list:
            for _ in range(num_randomizations):  # Randomize SMILES multiple times
                randomized_smiles = self.randomize_smiles(smiles)
                selfies = sf.encoder(randomized_smiles)
                tokens = sf.split_selfies(selfies)
                alphabet.update(tokens)  # Collect unique SELFIES tokens
            
        # Create a mapping from SELFIES tokens to unique Unicode characters
        base_unicode = 0x10000  # Start mapping from a non-overlapping point in Unicode
        for idx, token in enumerate(alphabet):
            unicode_char = chr(base_unicode + idx)
            self.unicode_mapping[token] = unicode_char
            self.reverse_unicode_mapping[unicode_char] = token
        
        return self.unicode_mapping

    def encode_selfies_to_unicode(self, selfies_sequence):
        """Convert SELFIES sequence to a string of Unicode characters."""
        tokens = sf.split_selfies(selfies_sequence)
        unicode_string = "".join(self.unicode_mapping[token] for token in tokens)
        return unicode_string

    def decode_unicode_to_selfies(self, unicode_string):
        """Convert Unicode string back to a SELFIES sequence using the reverse mapping."""
        selfies_tokens = [self.reverse_unicode_mapping[char] for char in unicode_string]
        selfies_string = "".join(selfies_tokens)  # Join back into a valid SELFIES string
        return selfies_string

    def process_data(self, file_path, vocab_size, tokenizer_path=None):
        def parse_smiles(file_path):
            smiles_list = []
            properties = []
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    smiles_list.append(row['SMILES'])
                    if self.props:
                        properties.append([float(value) for key, value in row.items() if key != 'SMILES'])  # Store properties
            return smiles_list, properties

        sequences, properties = parse_smiles(file_path)

        if tokenizer_path:
            # Load an existing tokenizer
            tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            # Generate SELFIES representations and create an alphabet
            # selfies_list = list(map(sf.encoder, sequences))
            self.create_selfies_alphabet(sequences)

            with open("models/unicode_mapping.json", "w") as f:
                json.dump(self.unicode_mapping, f)

            # Convert each SELFIES string to its Unicode representation
            unicode_selfies_strings = [self.encode_selfies_to_unicode(sf.encoder(smiles)) for smiles in sequences]

            # Merge all Unicode SELFIES strings into one long string separated by spaces
            merged_string = " ".join(unicode_selfies_strings)

            # Train BPE tokenizer on the Unicode-encoded SELFIES strings
            tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[EOS]"], vocab_size=vocab_size)
            tokenizer.train_from_iterator([merged_string], trainer)

            # Save tokenizer if desired
            tokenizer.save("models/selfies_tokenizer_final.json")

        return sequences, properties, tokenizer

    def encode(self, selfies_sequence):
        """Convert a SELFIES sequence to its Unicode representation and encode it using the tokenizer."""
        # Convert SELFIES to Unicode using the saved mapping
        unicode_string = self.encode_selfies_to_unicode((selfies_sequence))

        # Add special tokens [CLS] at the start and [EOS] at the end, no spaces in between
        encoded = self.tokenizer.encode(f"[CLS]{unicode_string}[EOS]")

        # Ensure token ids fit within the max length, pad if needed
        ids = encoded.ids + [self.tokenizer.token_to_id("[PAD]")] * (self.max_length - len(encoded.ids))
        ids = ids[:self.max_length]  # Truncate if it exceeds max_length

        return ids

    def decode(self, token_ids):
        """Convert token IDs back to the original SELFIES sequence."""
        # Get the token strings directly from token IDs (skip special tokens manually)
        tokens = [self.tokenizer.id_to_token(id) for id in token_ids if id not in [
            self.tokenizer.token_to_id("[PAD]"),
            self.tokenizer.token_to_id("[CLS]"),
            self.tokenizer.token_to_id("[EOS]"),
            self.tokenizer.token_to_id("[UNK]")
        ]]

        # Join tokens manually to avoid unwanted spaces
        decoded_output = "".join(tokens)  # Join without spaces

        # Convert Unicode back to SELFIES
        selfies_sequence = self.decode_unicode_to_selfies(decoded_output)
        
        return selfies_sequence


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
        # print(smiles)
        failed = False
        while True:
            try:
                # Randomize SMILES and convert to SELFIES
                # print(smiles)
                randomized_smiles = self.randomize_smiles(smiles)
                selfies_sequence = sf.encoder(randomized_smiles)  # Convert SMILES to SELFIES
                # print(selfies_sequence)
                # Encode the SELFIES sequence
                encoded_ids = self.encode(selfies_sequence)
                break
            except KeyError:
                failed = True
                break
            except Exception:
                continue
        
        if failed:
            return None, None

        # Return token ids and properties
        if self.props:
            return torch.tensor(encoded_ids, dtype=torch.long), torch.tensor(self.properties[idx], dtype=torch.float32)
        return torch.tensor(encoded_ids, dtype=torch.long), None
