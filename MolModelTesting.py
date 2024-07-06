import torch
from MolTransformer import MultiTaskTransformer
from tokenizers import Tokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import root_mean_squared_error, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

src_vocab_size = config["mol_model"]["src_vocab_size"]
tgt_vocab_size = config["mol_model"]["tgt_vocab_size"]
max_seq_length = config["mol_model"]["max_seq_length"]
num_tasks = config["mol_model"]["num_tasks"]
d_model = config["mol_model"]["d_model"]
num_heads = config["mol_model"]["num_heads"]
num_layers = config["mol_model"]["num_layers"]
d_ff = config["mol_model"]["d_ff"]
dropout = config["mol_model"]["dropout"]
learning_rate = config["mol_model"]["learning_rate"]
batch_size = config["mol_model"]["batch_size"]
device = config["mol_model"]["device"]
warmup_epochs = config["mol_model"]["warmup_epochs"]
total_epochs = config["mol_model"]["total_epochs"]
patience = config["mol_model"]["patience"]
pretrain_epochs = config["mol_model"]["pretrain_epochs"]
pretrain_learning_rate = config["mol_model"]["pretrain_learning_rate"]
tok_file = config["mol_model"]["tokenizer_file"]


model = MultiTaskTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks)
model.load_state_dict(torch.load('models/best_model.pt'))
# Ensure the model is on the correct device
model.to(device)
smiles_tokenizer = Tokenizer.from_file(tok_file)

def tokenize_smiles(smiles_string):
  # Initialize the dataset and dataloaders
  encoded = smiles_tokenizer.encode("[CLS] " + smiles_string + " [EOS]")
  ids = encoded.ids + [smiles_tokenizer.token_to_id("[PAD]")] * (128 - len(encoded.ids))
  ids = ids[:128]  # Ensure length does not exceed max_length
  return torch.tensor(ids, dtype=torch.long, device=device)

def bit2np(bitvector):
    bitstring = bitvector.ToBitString()
    intmap = map(int, bitstring)
    return np.array(list(intmap))

def extract_morgan(smiles, targets):
    X = []
    y = pd.DataFrame()
    for i, sm in enumerate(smiles):
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            print(sm)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) # Morgan (Similar to ECFP4)
        X.append(bit2np(fp))
        y = pd.concat([y, targets.iloc[[i]]], ignore_index=True)
    return np.array(X), y

def train_ecfp(data_path, target_columns, is_classification=False, n_repeats=5):
    data = pd.read_csv(data_path)
    data = data.dropna()

    # print(type(data[target_columns]))

    x_smiles = data['smiles'].to_numpy()
    X, y = extract_morgan(x_smiles, data[target_columns])
    # y = data[target_columns]

    # Initialize model
    keys = data.keys()[1:]
    if is_classification:
        model = MLPClassifier(max_iter=1000)
        metric = roc_auc_score
        metric_name = 'ROC-AUC'
    else:
        model = MLPRegressor(max_iter=1000)
        metric = root_mean_squared_error
        metric_name = 'RMSE'
    
    # Prepare for plotting
    train_sizes = [.0125, .025, .05, .1, .2, .4, .8]
    metrics = []
    metrics_std = []

    # Evaluate model performance for different train sizes
    for _, train_size in enumerate(tqdm(train_sizes, desc="Training Models")):
        overall = []
        for _ in range(n_repeats):
            total = np.empty(len(keys))
            for i in range(len(keys)):
                key = keys[i]
                y_cur = y[key].to_numpy()
                if(len(y_cur) * train_size <= 1):
                    continue
                X_train, X_test, y_train, y_test = train_test_split(X, y_cur, train_size=train_size, random_state=0, stratify=y_cur)
                model.fit(X_train, y_train)
                if is_classification:
                    y_pred = model.predict_proba(X_test)
                else:
                    y_pred = model.predict(X_test)

                if is_classification:
                    total[i] = metric(y_test, y_pred[:,1])
                else:
                    total[i] = metric(y_test, y_pred)

            overall.append(np.mean(total))
        metrics.append(np.mean(overall))
        metrics_std.append(np.std(overall))

    plt.errorbar(train_sizes, metrics, metrics_std, linestyle='dashed', color='orange', label='ECFP')
    # plt.plot(train_sizes, metrics, label='ecfp', color="red")



# Function to train and plot for a single dataset
def train_and_plot(data_path, target_columns, model, is_classification=False, n_repeats=5, title="Plot"):
    # Load dataset
    data = pd.read_csv(data_path)
    
    # Drop rows with missing values
    data = data.dropna()
    
    # Extract features and target
    x_smiles = data['smiles']
    x_tokenized = x_smiles.apply(tokenize_smiles)
    # run through model
    x_tokenized = x_tokenized.to_numpy()
    for i in tqdm(range(len(x_tokenized))):
        x_tokenized[i] = model.encode_smiles(x_tokenized[i].reshape(1,-1))[1].cpu().detach().numpy()
    X = np.stack((x_tokenized)).reshape(-1, 768)
    y = data[target_columns]

    keys = data.keys()[1:]
    if is_classification:
        model = MLPClassifier(max_iter=1000)
        metric = roc_auc_score
        metric_name = 'ROC-AUC'
    else:
        model = MLPRegressor(max_iter=1000)
        metric = root_mean_squared_error
        metric_name = 'RMSE'
    
    # Prepare for plotting
    train_sizes = [.0125, .025, .05, .1, .2, .4, .8]
    metrics = []
    metrics_std = []

    # Evaluate model performance for different train sizes
    for _, train_size in enumerate(tqdm(train_sizes, desc="Training Models")):
        overall = []
        for _ in range(n_repeats):
            total = np.empty(len(keys))
            for i in range(len(keys)):
                key = keys[i]
                y_cur = y[key].to_numpy()
                if(len(y_cur) * train_size <= 1):
                    continue
                X_train, X_test, y_train, y_test = train_test_split(X, y_cur, train_size=train_size, random_state=0, stratify=y_cur)
                model.fit(X_train, y_train)
                if is_classification:
                    y_pred = model.predict_proba(X_test)
                else:
                    y_pred = model.predict(X_test)

                if is_classification:
                    total[i] = metric(y_test, y_pred[:,1])
                else:
                    total[i] = metric(y_test, y_pred)

            overall.append(np.mean(total))
        metrics.append(np.mean(overall))
        metrics_std.append(np.std(overall))

    plt.errorbar(train_sizes, metrics, metrics_std, linestyle='dashdot', color='purple', label='Transformer')
    
    # Plotting
    plt.xlabel('Train size')
    plt.ylabel(metric_name)
    # Set x-axis to log scale
    plt.xscale('log')

    # Manually set the ticks for the x-axis
    plt.xticks([0.025, 0.05, 0.1, 0.2, 0.4, 0.8], ['0.025', '0.05', '0.1', '0.2', '0.4', '0.8'])
    plt.title(f'{title}')
    # plt.plot(train_sizes, metrics, label=f'Transformer', color="blue")

# Example usage for each dataset
# Train and plot for each relevant CSV file

datasets_info = [
    ('mol_net_datasets/clintox_relevant.csv', ['FDA_APPROVED', 'CT_TOX'], 'ClinTox', True),
    ('mol_net_datasets/sider_relevant.csv', ['Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues', 
                                     'Eye disorders', 'Investigations', 'Musculoskeletal and connective tissue disorders', 
                                     'Gastrointestinal disorders', 'Social circumstances', 'Immune system disorders', 
                                     'Reproductive system and breast disorders'], 'SIDER', True),
    ('mol_net_datasets/tox21_relevant.csv', ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
                                     'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'], 'Tox21', True),
    ('mol_net_datasets/bbbp_relevant.csv', ['p_np'], 'BBBP', True),
    ('mol_net_datasets/bace_relevant.csv', ['Class'], 'BACE', True),
    ('mol_net_datasets/hiv_relevant.csv', ['HIV_active'], 'HIV', True),
    ('mol_net_datasets/lipophilicity_relevant.csv', ['exp'], 'Lipophilicity', False),
    ('mol_net_datasets/free_solv_relevant.csv', ['expt'], 'FreeSolv', False),
    ('mol_net_datasets/esol_relevant.csv', ['ESOL predicted log solubility in mols per litre'], 'ESOL', False)
]

# Create subplots

# Loop through each dataset info and create a plot
for i, (data_path, target_columns, title, is_classification) in enumerate(datasets_info):
    plt.figure(figsize=([10,8]))
    train_and_plot(data_path, target_columns, model, is_classification, 5, title)
    train_ecfp(data_path, target_columns, is_classification, n_repeats=5)
    print("FINISHED")
    plt.legend(loc="upper right")
    plt.savefig(f"mol_net_datasets/images/{title}.png")
    plt.clf()