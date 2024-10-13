from model_property2 import *
from data_preparation import *
from model_property2 import SentenceVAE
from params import *
from multiprocessing import cpu_count
import numpy as np
import pickle
from util import *
import pickle 
import os
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import argparse
import json
from collections import OrderedDict
from torch.utils.data import random_split, DataLoader
import importlib.util
import yaml

# Load the YAML configuration file
with open("params.yaml", "r") as file:
    Args = yaml.safe_load(file)

print(f'Loaded parameters: {Args}')


print('############### data loading ######################')
#data_dir = 'data/'
#data_list = ['zinc250k/zinc_train', 'zinc250k/zinc_test']

# dataset = SMILESDataset(file_path, vocab_size=79, max_length=128, tokenizer_path="models/selfies_tok.json")

# data_train, data_valid = text2dict_zinc(Args['data_list'])
#txt, data_train_ = smi_postprocessing(data_train,biology_flag=False,LB_smi=15,UB_smi=120)

# print("Number of training SMILES >>>> " , len(data_train['SMILES']))
# print("Number of validation SMILES >>>> ", len(data_valid['SMILES']))

# smiles preprocessing:
'''
We filter the SMILES based on two criterias:
1) 8 < logP < -3 and 600 < MolecularWeight < 50,
2) 120 < smiles with length <  20

 pre-processing step is happening in smi_postprocessing is function.
 
 Please note that, biology_flag is served for the situations that the user has an access to biological label.
####
'''
biology_flag = False
#LogP = [-3,8]
#MW = [50,600]
#LB_smi= 20
# UB_smi = 120
#txt,data_train_ = smi_postprocessing(data_train,biology_flag,LB_smi,UB_smi,LogP,MW)
# txt, data_train_ = smi_postprocessing(data_train, biology_flag, UB_smi)
# _, data_valid_ = smi_postprocessing(data_valid,biology_flag,UB_smi)
# print(len(data_train_['SMILES']), len(data_train_['removed']))
# print(len(data_valid_['SMILES']), len(data_valid_['removed']))

'''
Dictionary building part: 
In this part, we generate dictionary based on the output of the smi_postrocessing function.

Then, we shape the data to the usable format for the pytorch using dataset_building function,
followed by DataLoader function.

'''
data_type = 'pure_smiles'
# char2ind,ind2char,sos_indx,eos_indx,pad_indx = dictionary_build(txt)
#max_sequence_length = np.max(np.asarray([len(smi) for smi in data_train_['SMILES']]))
max_sequence_length = 600

dataset = DatasetBuilding("../data/smiles_10000_selected_features_cleaned.csv", max_sequence_length, tokenizer="../models/selfies_tok.json")

train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing
dataset_train, dataset_valid = random_split(dataset, [train_size, test_size])

# dataset_train = DatasetBuilding(char2ind,data_train_,max_sequence_length,data_type)
dataloader_train = DataLoader(dataset=dataset_train, batch_size= Args['batch_size'], shuffle = True)
# dataset_valid = DatasetBuilding(char2ind,data_valid_,max_sequence_length,data_type)
dataloader_valid = DataLoader(dataset = dataset_valid, batch_size = Args['batch_size'], shuffle = False)

print(len(dataset_train), len(dataset_valid))

'''
Model loading step: 

We defined the model with pvae fuunction. reader for more detail of structure of network is encoraged to visit pvae.py.

'''
vocab_size = 79
sos_indx = 2
eos_indx = 3
pad_indx = 0
'''
model = pvae(vocab_size, Args.embedding_size, Args.rnn_type, Args.hidden_size, Args.word_dropout, Args.latent_size,
                    sos_indx, eos_indx, pad_indx, max_sequence_length,Args.nr_classes,Args.device_id,
                    num_layers=1,bidirectional=False, gpu_exist = Args.gpu_exist)
'''
print('Latent_size:', Args['latent_size'])
model = SentenceVAE(vocab_size, Args['embedding_size'], Args['rnn_type'], Args['hidden_size'], Args['word_dropout'], Args['latent_size'],
                    sos_indx, eos_indx, pad_indx, max_sequence_length,Args['device_id'],
                    num_layers=1, nr_prop = Args['nr_classes'],bidirectional=False, gpu_exist = Args['gpu_exist'])
if torch.cuda.is_available():
    torch.cuda.set_device(Args['device_id'])
    model = model.cuda(Args['device_id'])

tokenizer = SelfiesTok.load("../models/selfies_tok.json")
class_weight = char_weight(dataset,tokenizer.token_to_id, tokenizer)
class_weight = torch.FloatTensor(class_weight).cuda(device=Args['device_id'])

optimizer = torch.optim.Adam(model.parameters(), lr = Args['learning_rate'])
NLL = torch.nn.CrossEntropyLoss( weight = class_weight, reduction= 'sum', ignore_index=pad_indx)

step = 0
from loss_vae import *
tracker = defaultdict(list)
for epoch in range(0, Args['epochs']):
    
    optimizer,lr = adjust_learning_rate(optimizer, epoch, Args['learning_rate'])
    temp = defaultdict(list)
    score_acc = []
    AUC_acc = []
    
    for iteration, batch in enumerate(dataloader_train):
        
        batch = batch2tensor(batch,Args)
        batch_size = batch['input'].size(0)
        model.train()
        ######     Forward pass  ######
        logp, mean, logv, z = model(batch['input'], batch['length'])
       
       
        NLL_loss, KL_loss, KL_weight = loss_fn(NLL,logp, batch['target'],batch['length'], mean, logv,
                                                               Args['anneal_function'], step, Args['k0'],Args['x0'])
        loss = (NLL_loss + KL_weight * KL_loss)/batch_size
   
        #### Evaluation #####  
        temp['NLL_loss'].append(NLL_loss.item()/batch_size)
        temp['KL_loss'].append(KL_weight*KL_loss.item()/batch_size)
        temp['ELBO'].append(loss.item())
        
        #### Backward pass #####

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1 # for anneling


    #### Validation
    total_reconstructed = 0
    total_sequences = 10  # We will check the first 10 sequences in the validation set
    completely_correct_sequences = 0  # Count sequences that are fully correct
    for iteration, batch in enumerate(dataloader_valid):
        batch = batch2tensor(batch, Args)
        batch_size = batch['input'].size(0) 
        
        logp, mean, logv, z = model(batch['input'], batch['length'])

        NLL_loss, KL_loss, KL_weight = loss_fn(NLL, logp, batch['target'], batch['length'], mean, logv,
                                                               Args['anneal_function'], step, Args['k0'], Args['x0'])
        loss = (NLL_loss + KL_weight * KL_loss) / batch_size

        #### Evaluation #####
        temp['val_NLL_loss'].append(NLL_loss.item() / batch_size)
        temp['val_KL_loss'].append(KL_weight * KL_loss.item() / batch_size)
        temp['val_ELBO'].append(loss.item())


        # Reconstruction accuracy for the first 10 sequences
        if iteration == 0:  # Only process the first batch
            predictions = torch.argmax(logp, dim=-1)  # Get the predicted tokens
            targets = batch['target']  # Actual targets

            # Ignore padding tokens (pad_indx)
            mask = targets != pad_indx  # Mask out padding tokens

            # Calculate accuracy ignoring padding tokens
            correct_tokens = (predictions == targets) & mask  # Correct tokens ignoring padding
            token_accuracy = correct_tokens.float().sum() / mask.float().sum()  # Token-level accuracy

            # Check for completely correct sequences (where all tokens in a sequence are correct)
            for i in range(total_sequences):
                if torch.all(correct_tokens[i]):  # Check if all tokens are correct in the sequence
                    completely_correct_sequences += 1

            total_reconstructed = token_accuracy.item()  # Token-level reconstruction accuracy

    # Calculate and print the reconstruction accuracy and number of fully correct sequences
    reconstruction_accuracy = total_reconstructed * 100  # Convert to percentage
    print(f"Reconstruction accuracy for first 10 validation sequences at epoch {epoch}: {reconstruction_accuracy:.2f}%")
    print(f"Number of completely correct sequences: {completely_correct_sequences} out of {total_sequences}")



   # for iteration, batch in enumerate(dataloader_test):    
    #    batch = batch2tensor(batch,Args)
     #   batch_size = batch['input'].size(0)
      #  model.train()
        ######     Forward pass  ######
       # logp, mean, logv, z,prediction = model(batch['input'], batch['length'])
        #model.eval()

    tracker['NLL_loss'].append(np.mean(np.asarray(temp['NLL_loss'])))
    tracker['KL_loss'].append(np.mean(np.asarray(temp['KL_loss'])))
    tracker['ELBO'].append(np.mean(np.asarray(temp['ELBO'])))

    tracker['val_NLL_loss'].append(np.mean(np.asarray(temp['val_NLL_loss'])))
    tracker['val_KL_loss'].append(np.mean(np.asarray(temp['val_KL_loss'])))
    tracker['val_ELBO'].append(np.mean(np.asarray(temp['val_ELBO'])))

    print(" epoch %04d/%i, 'ELBO' %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f, lr %9.7f"
          %(epoch,Args['epochs'], 
            np.mean(np.asarray(tracker['ELBO'])),
            np.mean(np.asarray(tracker['NLL_loss'])),
            np.mean(np.asarray(tracker['KL_loss'])),    
            KL_weight,
            lr))

    print("\t\t  'val_ELBO' %9.4f, val_NLL-Loss %9.4f, val_KL-Loss %9.4f"
          % (np.mean(np.asarray(tracker['val_ELBO'])),
             np.mean(np.asarray(tracker['val_NLL_loss'])),
             np.mean(np.asarray(tracker['val_KL_loss']))))
   
    
    if epoch % Args['save_every'] == 0:
        checkpoint_path = os.path.join(Args['save_dir'], "E%i.pytorch"%(epoch))
        torch.save(model.state_dict(), checkpoint_path)