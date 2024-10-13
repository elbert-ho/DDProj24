#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SELFIES: a robust representation of semantically constrained graphs with an
    example application in chemistry (https://arxiv.org/abs/1905.13741)
    by Mario Krenn, Florian Haese, AkshatKuman Nigam, Pascal Friederich,
    Alan Aspuru-Guzik.

    Variational Autoencoder (VAE) for chemistry
        comparing SMILES and SELFIES representation using reconstruction
        quality, diversity and latent space validity as metrics of
        interest

information:
    ML framework: pytorch
    chemistry framework: RDKit

    get_selfie_and_smiles_encodings_for_dataset
        generate complete encoding (inclusive alphabet) for SMILES and
        SELFIES given a data file

    VAEEncoder
        fully connected, 3 layer neural network - encodes a one-hot
        representation of molecule (in SMILES or SELFIES representation)
        to latent space

    VAEDecoder
        decodes point in latent space using an RNN

    latent_space_quality
        samples points from latent space, decodes them into molecules,
        calculates chemical validity (using RDKit's MolFromSmiles), calculates
        diversity
"""

import os
import sys
import time
import json

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import rdBase
from rdkit.Chem import MolFromSmiles
from torch import nn
from rdkit import Chem


import selfies as sf
from data_loader import \
    multiple_selfies_to_hot, multiple_smile_to_hot

rdBase.DisableLog('rdApp.error')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_dir(directory):
    os.makedirs(directory)


def save_models(encoder, decoder, epoch):
    out_dir = './saved_models/{}'.format(epoch)
    _make_dir(out_dir)
    torch.save(encoder, '{}/E'.format(out_dir))
    torch.save(decoder, '{}/D'.format(out_dir))


class VAEEncoder(nn.Module):

    def __init__(self, in_dimension, layer_1d, layer_2d, layer_3d,
                 latent_dimension):
        """
        Fully Connected layers to encode molecule to latent space
        """
        super(VAEEncoder, self).__init__()
        self.latent_dimension = latent_dimension

        # Reduce dimension up to second last layer of Encoder
        self.encode_nn = nn.Sequential(
            nn.Linear(in_dimension, layer_1d),
            nn.ReLU(),
            nn.Linear(layer_1d, layer_2d),
            nn.ReLU(),
            nn.Linear(layer_2d, layer_3d),
            nn.ReLU()
        )

        # print(latent_dimension)
        # print(layer_1d)
        # print(layer_2d)
        # print(layer_3d)

        # Latent space mean
        self.encode_mu = nn.Linear(layer_3d, latent_dimension)

        # Latent space variance
        self.encode_log_var = nn.Linear(layer_3d, latent_dimension)

    @staticmethod
    def reparameterize(mu, log_var):
        """
        This trick is explained well here:
            https://stats.stackexchange.com/a/16338
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """
        Pass throught the Encoder
        """
        # Get results of encoder network
        h1 = self.encode_nn(x)

        # latent space
        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # print(mu)
        # print(log_var)

        return z, mu, log_var


class VAEDecoder(nn.Module):

    def __init__(self, latent_dimension, gru_stack_size, gru_neurons_num,
                 out_dimension):
        """
        Through Decoder
        """
        super(VAEDecoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.gru_stack_size = gru_stack_size
        self.gru_neurons_num = gru_neurons_num

        # Simple Decoder
        self.decode_RNN = nn.GRU(
            input_size=latent_dimension,
            hidden_size=gru_neurons_num,
            num_layers=gru_stack_size,
            batch_first=False)

        self.decode_FC = nn.Sequential(
            nn.Linear(gru_neurons_num, out_dimension),
        )

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters())
        return weight.new_zeros(self.gru_stack_size, batch_size,
                                self.gru_neurons_num)

    def forward(self, z, hidden):
        """
        A forward pass throught the entire model.
        """

        # Decode
        l1, hidden = self.decode_RNN(z, hidden)
        decoded = self.decode_FC(l1)  # fully connected layer

        return decoded, hidden


def is_correct_smiles(smiles):
    """
    Using RDKit to calculate whether molecule is syntactically and
    semantically valid.
    """
    if smiles == "":
        return False

    try:
        return MolFromSmiles(smiles, sanitize=True) is not None
    except Exception:
        return False


def sample_latent_space(vae_encoder, vae_decoder, sample_len):
    vae_encoder.eval()
    vae_decoder.eval()

    gathered_atoms = []

    fancy_latent_point = torch.randn(1, 1, vae_encoder.latent_dimension,
                                     device=device)
    hidden = vae_decoder.init_hidden()

    # runs over letters from molecules (len=size of largest molecule)
    for _ in range(sample_len):
        out_one_hot, hidden = vae_decoder(fancy_latent_point, hidden)

        out_one_hot = out_one_hot.flatten().detach()
        soft = nn.Softmax(0)
        out_one_hot = soft(out_one_hot)

        out_index = out_one_hot.argmax(0)
        gathered_atoms.append(out_index.data.cpu().tolist())

    vae_encoder.train()
    vae_decoder.train()

    return gathered_atoms


def latent_space_quality(vae_encoder, vae_decoder, type_of_encoding,
                         alphabet, sample_num, sample_len):
    total_correct = 0
    all_correct_molecules = set()
    print(f"latent_space_quality:"
          f" Take {sample_num} samples from the latent space")

    for _ in range(1, sample_num + 1):

        molecule_pre = ''
        for i in sample_latent_space(vae_encoder, vae_decoder, sample_len):
            molecule_pre += alphabet[i]
        molecule = molecule_pre.replace(' ', '')
        molecule = molecule_pre.replace('[EOS]', '')

        if type_of_encoding == 1:  # if SELFIES, decode to SMILES
            molecule = sf.decoder(molecule)

        if is_correct_smiles(molecule):
            total_correct += 1
            all_correct_molecules.add(molecule)

    return total_correct, len(all_correct_molecules)


def cyclical_annealing(T, M, step, R=0.4, max_kl_weight=1):
    """
    Implementing: <https://arxiv.org/abs/1903.10145>
    T = Total steps 
    M = Number of cycles 
    R = Proportion used to increase beta
    step = Global step 
    """
    period = T / M  # Length of one cycle
    internal_period = step % period  # Position within the current cycle
    tau = internal_period / period  # Normalize to [0, 1] range within the cycle

    if tau <= R:  # Linearly increase phase
        kl_weight = min(max_kl_weight, tau / R * max_kl_weight)
    else:  # Sudden drop phase
        kl_weight = 0

    return kl_weight

def quality_in_valid_set(vae_encoder, vae_decoder, data_valid, batch_size, padding_idx):
    data_valid = data_valid[torch.randperm(data_valid.size()[0])]  # shuffle
    num_batches_valid = len(data_valid) // batch_size

    quality_list = []
    exact_match_list = []
    for batch_iteration in range(min(25, num_batches_valid)):

        # get batch
        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        batch = data_valid[start_idx: stop_idx]
        _, trg_len, _ = batch.size()

        inp_flat_one_hot = batch.flatten(start_dim=1)
        latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)

        latent_points = latent_points.unsqueeze(0)
        hidden = vae_decoder.init_hidden(batch_size=batch_size)
        out_one_hot = torch.zeros_like(batch, device=device)
        eos_token_index = 1
        for seq_index in range(trg_len):
            out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
            out_one_hot[:, seq_index, :] = out_one_hot_line[0]

            # Check for EOS token in each sequence in the batch
            eos_indices = (out_one_hot_line.argmax(dim=-1) == eos_token_index)
            
            # If all sequences in the batch have reached the EOS token, break the loop
            if eos_indices.all():
                break

        # assess reconstruction quality
        quality = compute_recon_quality(batch, out_one_hot, padding_idx)
        quality_list.append(quality)

        # assess exact match for validation
        exact_match = compute_exact_match_quality(batch, out_one_hot)
        exact_match_list.append(exact_match)

    return np.mean(quality_list).item(), np.mean(exact_match_list).item()

def decode_one_hot_to_smiles(one_hot_tensor, alphabet):
    """
    Convert a one-hot encoded tensor back to a SMILES string using the provided alphabet.
    """
    indices = one_hot_tensor.argmax(dim=-1).squeeze().tolist()
    smiles = ''.join([alphabet[i] for i in indices if i < len(alphabet)])
    return smiles.strip()


def train_model(vae_encoder, vae_decoder,
                data_train, data_valid, num_epochs, batch_size,
                lr_enc, lr_dec, T, M, sample_num, sample_len, alphabet, type_of_encoding, padding_idx):
    """
    Train the Variational Auto-Encoder
    """

    print('num_epochs: ', num_epochs)

    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam(vae_encoder.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(vae_decoder.parameters(), lr=lr_dec)

    data_train = data_train.clone().detach().to(device)
    num_batches_train = int(len(data_train) / batch_size)

    quality_valid_list = [0, 0, 0, 0]
    global_step = 0

    for epoch in range(num_epochs):

        data_train = data_train[torch.randperm(data_train.size()[0])]

        start = time.time()
        for batch_iteration in range(num_batches_train):  # batch iterator

            # Compute the current value of the KL β-term using the cyclical annealing schedule
            # KLD_alpha = cyclical_annealing(T, M, global_step)
            KLD_alpha = 1e-5
            # KLD_alpha = 0

            # manual batch iterations
            start_idx = batch_iteration * batch_size
            stop_idx = (batch_iteration + 1) * batch_size
            batch = data_train[start_idx: stop_idx]

            # reshaping for efficient parallelization
            inp_flat_one_hot = batch.flatten(start_dim=1)
            latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)

            # initialization hidden internal state of RNN (RNN has two inputs
            # and two outputs:)
            #    input: latent space & hidden state
            #    output: one-hot encoding of one character of molecule & hidden
            #    state the hidden state acts as the internal memory
            latent_points = latent_points.unsqueeze(0)
            hidden = vae_decoder.init_hidden(batch_size=batch_size)

            # decoding from RNN N times, where N is the length of the largest
            # molecule (all molecules are padded)
            out_one_hot = torch.zeros_like(batch, device=device)
            eos_token_index = 1
            for seq_index in range(batch.shape[1]):
                out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
                out_one_hot[:, seq_index, :] = out_one_hot_line[0]

                # Check for EOS token in each sequence in the batch
                eos_indices = (out_one_hot_line.argmax(dim=-1) == eos_token_index)
                
                # If all sequences in the batch have reached the EOS token, break the loop
                if eos_indices.all():
                    break



            # compute ELBO
            loss, recon_loss, kld_loss = compute_elbo(batch, out_one_hot, mus, log_vars, KLD_alpha, padding_idx=padding_idx)
            # perform back propogation
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(vae_decoder.parameters(), 0.5)
            optimizer_encoder.step()
            optimizer_decoder.step()
            global_step += 1
            if batch_iteration % 30 == 0:
                end = time.time()

                # assess reconstruction quality
                quality_train = compute_recon_quality(batch, out_one_hot, padding_idx=padding_idx)
                quality_valid, exact_match_valid = quality_in_valid_set(vae_encoder, vae_decoder,
                                                     data_valid, batch_size, padding_idx=padding_idx)

                report = 'Epoch: %d,  Batch: %d / %d,\t(loss: %.4f\t| ' \
                         'quality: %.4f | quality_valid: %.4f | exact_match_valid: %.4f)\t' \
                         'ELAPSED TIME: %.5f' \
                         % (epoch, batch_iteration, num_batches_train,
                            loss.item(), quality_train, quality_valid, exact_match_valid,
                            end - start)
                print(report)
                print(f"Reconstruction Loss: {recon_loss.item()}, KLD Loss: {kld_loss.item()}, BETA: {KLD_alpha}")
                start = time.time()

        print("\nValidation set reconstruction examples:")
        with torch.no_grad():
            for i in range(min(5, len(data_valid))):
                original = data_valid[i].unsqueeze(0).to(device)
                original_flat_one_hot = original.flatten(start_dim=1)
                latent_point, _, _ = vae_encoder(original_flat_one_hot)
                latent_point = latent_point.unsqueeze(0)
                hidden = vae_decoder.init_hidden(batch_size=1)
                
                decoded_one_hot = torch.zeros_like(original, device=device)
                eos_token_index = 1
                for seq_index in range(original.shape[1]):
                    decoded_line, hidden = vae_decoder(latent_point, hidden)
                    decoded_one_hot[:, seq_index, :] = decoded_line[0]

                    # Check for EOS token in each sequence in the batch
                    eos_indices = (decoded_line.argmax(dim=-1) == eos_token_index)
                    
                    # If all sequences in the batch have reached the EOS token, break the loop
                    if eos_indices.all():
                        break


                original_smiles = decode_one_hot_to_smiles(original, alphabet)
                reconstructed_smiles = decode_one_hot_to_smiles(decoded_one_hot, alphabet)

                print(f"Original: {original_smiles}")
                print(f"Reconstructed: {reconstructed_smiles}\n")

        quality_valid, _ = quality_in_valid_set(vae_encoder, vae_decoder,
                                             data_valid, batch_size, padding_idx=padding_idx)
        quality_valid_list.append(quality_valid)

        # only measure validity of reconstruction improved
        quality_increase = len(quality_valid_list) \
                           - np.argmax(quality_valid_list)
        if quality_increase == 1 and quality_valid_list[-1] > 50.:
            corr, unique = latent_space_quality(vae_encoder, vae_decoder,
                                                type_of_encoding, alphabet,
                                                sample_num, sample_len)
        else:
            corr, unique = -1., -1.

        report = 'Validity: %.5f %% | Diversity: %.5f %% | ' \
                 'Reconstruction: %.5f %%' \
                 % (corr * 100. / sample_num, unique * 100. / sample_num,
                    quality_valid)
        print(report)

        with open('results.dat', 'a') as content:
            content.write(report + '\n')

        if quality_valid_list[-1] < 70. and epoch > 200:
            break

        if quality_increase > 20:
            save_models(vae_encoder, vae_decoder, epoch)
            print('Early stopping criteria')
            break


def compute_elbo(x, x_hat, mus, log_vars, KLD_alpha, padding_idx=None):
    """
    Compute the Evidence Lower Bound (ELBO) for the VAE, excluding padding tokens.
    """
    inp = x_hat.reshape(-1, x_hat.shape[2])
    target = x.reshape(-1, x.shape[2]).argmax(1)

    if padding_idx is not None:
        mask = (target != padding_idx)  # Mask to exclude padding tokens
        inp = inp[mask]
        target = target[mask]

    criterion = torch.nn.CrossEntropyLoss()
    recon_loss = criterion(inp, target)
    kld = -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp())

    return recon_loss + KLD_alpha * kld, recon_loss, kld



def compute_recon_quality(x, x_hat, padding_idx=None):
    """
    Compute the reconstruction quality while ignoring padding tokens.
    """
    x_indices = x.reshape(-1, x.shape[2]).argmax(1)
    x_hat_indices = x_hat.reshape(-1, x_hat.shape[2]).argmax(1)

    if padding_idx is not None:
        mask = (x_indices != padding_idx)  # Mask to exclude padding tokens
        x_indices = x_indices[mask]
        x_hat_indices = x_hat_indices[mask]

    differences = 1. - torch.abs(x_hat_indices - x_indices)
    differences = torch.clamp(differences, min=0., max=1.).double()
    quality = 100. * torch.mean(differences)
    quality = quality.detach().cpu().numpy()

    return quality


def compute_exact_match_quality(x, x_hat):
    """
    Compute the percentage of sequences that are completely correct.
    """
    x_indices = x.reshape(x.shape[0], -1, x.shape[2]).argmax(2)
    x_hat_indices = x_hat.reshape(x_hat.shape[0], -1, x_hat.shape[2]).argmax(2)

    # Check if the entire sequence is exactly the same
    exact_matches = torch.all(x_indices == x_hat_indices, dim=1)
    exact_match_percentage = 100. * torch.mean(exact_matches.float())
    
    return exact_match_percentage.detach().cpu().numpy()

def randomize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    randomized_smiles = Chem.MolToSmiles(mol, doRandom=True)
    return randomized_smiles

def get_selfie_and_smiles_encodings_for_dataset(file_path):
    """
    Returns encoding, alphabet and length of largest molecule in SMILES and
    SELFIES, given a file containing SMILES molecules.

    input:
        csv file with molecules. Column's name must be 'smiles'.
    output:
        - selfies encoding
        - selfies alphabet
        - longest selfies string
        - smiles encoding (equivalent to file content)
        - smiles alphabet (character based)
        - longest smiles string
    """

    df = pd.read_csv(file_path)

    smiles_list = np.asanyarray(df["SMILES"])

    smiles_alphabet = list(set(''.join(smiles_list)))
    smiles_alphabet.append(' ')  # for padding

    largest_smiles_len = len(max(smiles_list, key=len))

    print('--> Translating SMILES to SELFIES...')
    selfies_list = list(map(sf.encoder, smiles_list))

    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    # selfies_alphabet = list(all_selfies_symbols)
    
    # selfies_alphabet.append("[nop]")

    selfies_alphabet = ['[nop]', '[EOS]'] + list(all_selfies_symbols)

    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)

    print('Finished translating SMILES to SELFIES.')

    return selfies_list, selfies_alphabet, largest_selfies_len, \
           smiles_list, smiles_alphabet, largest_smiles_len


def save_tokenizers(selfies_alphabet, selfies_tokenizer_path):
    """
    Save the tokenizers (SMILES and SELFIES alphabets) to files.
    """    
    # Save SELFIES alphabet as a JSON file
    with open(selfies_tokenizer_path, 'w') as f:
        json.dump(selfies_alphabet, f)
    
    print(f"SELFIES tokenizer saved to {selfies_tokenizer_path}")

def main():
    content = open('logfile.dat', 'w')
    content.close()
    content = open('results.dat', 'w')
    content.close()

    if os.path.exists("settings2.yml"):
        settings = yaml.safe_load(open("settings2.yml", "r"))
    else:
        print("Expected a file settings2.yml but didn't find it.")
        return

    print('--> Acquiring data...')
    type_of_encoding = settings['data']['type_of_encoding']
    file_name_smiles = settings['data']['smiles_file']

    print('Finished acquiring data.')

    if type_of_encoding == 0:
        print('Representation: SMILES')
        _, _, _, encoding_list, encoding_alphabet, largest_molecule_len = \
            get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)

        print('--> Creating one-hot encoding...')
        data = multiple_smile_to_hot(encoding_list, largest_molecule_len,
                                     encoding_alphabet)
        print('Finished creating one-hot encoding.')

    elif type_of_encoding == 1:
        print('Representation: SELFIES')
        encoding_list, encoding_alphabet, largest_molecule_len, _, _, _ = \
            get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)

        print('--> Creating one-hot encoding...')
        data = multiple_selfies_to_hot(encoding_list, largest_molecule_len,
                                       encoding_alphabet)
        print('Finished creating one-hot encoding.')

    else:
        print("type_of_encoding not in {0, 1}.")
        return

    len_max_molec = data.shape[1]
    len_alphabet = data.shape[2]
    len_max_mol_one_hot = len_max_molec * len_alphabet

    print(' ')
    print(f"Alphabet has {len_alphabet} letters, "
          f"largest molecule is {len_max_molec} letters.")

    # print(encoding_alphabet)
    # print(encoding_alphabet[len(encoding_alphabet) - 1])
    # exit()

    selfies_tokenizer_path = "data/selfies_tokenizer_final.json"
    save_tokenizers(encoding_alphabet, selfies_tokenizer_path)

    data_parameters = settings['data']
    batch_size = data_parameters['batch_size']

    encoder_parameter = settings['encoder']
    decoder_parameter = settings['decoder']
    training_parameters = settings['training']

    vae_encoder = VAEEncoder(in_dimension=len_max_mol_one_hot,
                             **encoder_parameter).to(device)
    vae_decoder = VAEDecoder(**decoder_parameter,
                             out_dimension=len(encoding_alphabet)).to(device)

    print('*' * 15, ': -->', device)

    data = torch.tensor(data, dtype=torch.float).to(device)

    train_valid_test_size = [0.75, 0.25, 0.0]
    data = data[torch.randperm(data.size()[0])]
    idx_train_val = int(len(data) * train_valid_test_size[0])
    idx_val_test = idx_train_val + int(len(data) * train_valid_test_size[1])

    data_train = data[0:idx_train_val]
    data_valid = data[idx_train_val:idx_val_test]

    print("start training")
    train_model(**training_parameters,
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            batch_size=batch_size,
            data_train=data_train,
            data_valid=data_valid,
            T=160000,       # Total steps for cyclical annealing
            M=640,       # Number of cycles for cyclical annealing
            alphabet=encoding_alphabet,
            type_of_encoding=type_of_encoding,
            sample_len=len_max_molec,
            padding_idx=0)


    with open('COMPLETED', 'w') as content:
        content.write('exit code: 0')


if __name__ == '__main__':
    try:
        main()
    except AttributeError:
        _, error_message, _ = sys.exc_info()
        print(error_message)
