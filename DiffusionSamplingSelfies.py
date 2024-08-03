import torch
from DiffusionModelGLIDE import *
import yaml
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from unet_condition import Text2ImUNet
from ProtLigDataset import ProtLigDataset
from MolPropModel import PropertyModel
from transformers import EsmTokenizer, EsmModel
from tokenizers import Tokenizer
from MolTransformerSelfies import MultiTaskTransformer
from rdkit import Chem
from rdkit.Chem import Draw
from pIC50Predictor import pIC50Predictor
from SelfiesTok import SelfiesTok
import selfies as sf

# FIX
def get_balanced_grad(x, t, prot):
    # x size: [1, 1, 32768]
    # prot size: [1, 1280]
    # t size: [1, 1]
    pic50_model.train()
    qed_model.train()
    sas_model.train()

    x.requires_grad = True
    eps = 1e-8
    g_t = torch.zeros(3, *x.shape, device=device)  # Smoothed gradients
    pic50_pred = pic50_model(x.squeeze(1), prot, t)
    qed_pred = qed_model(x.squeeze(1), t).clamp(1, 10)
    sas_pred = sas_model(x.squeeze(1), t).clamp(0, 1)
    g_t[0] = torch.autograd.grad(pic50_pred, x.squeeze(1))[0]
    g_t[1] = torch.autograd.grad(qed_pred, x.squeeze(1))[0]
    g_t[2] = -torch.autograd.grad(sas_pred, x.squeeze(1))[0]

    g_norms = torch.linalg.vector_norm(g_t, dim=1)
    weights = torch.max(g_norms) / (g_norms + eps).unsqueeze(1)
    g_k = torch.sum(weights * g_t, dim=0).unsqueeze(1)
    return g_k

with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_diff_step = config["diffusion_model"]["num_diffusion_steps"]
batch_size = config["diffusion_model"]["batch_size"]
protein_embedding_dim = config["protein_model"]["protein_embedding_dim"]
lr = config["diffusion_model"]["lr"]
epochs = config["diffusion_model"]["epochs"]
patience = config["diffusion_model"]["patience"]
time_embed_dim = config["prop_model"]["time_embed_dim"]
src_vocab_size = config["mol_model"]["src_vocab_size"]
tgt_vocab_size = config["mol_model"]["tgt_vocab_size"]
max_seq_length = config["mol_model"]["max_seq_length"]
num_tasks = config["mol_model"]["num_tasks"]
d_model = config["mol_model"]["d_model"]
num_heads = config["mol_model"]["num_heads"]
num_layers = config["mol_model"]["num_layers"]
d_ff = config["mol_model"]["d_ff"]
dropout = config["mol_model"]["dropout"]
tok_file = config["mol_model"]["tokenizer_file"]
mol_model = MultiTaskTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks).to(device)
mol_model.load_state_dict(torch.load('models/selfies_transformer.pt', map_location=device))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = config["mol_model"]["d_model"] * config["mol_model"]["max_seq_length"]
qed_model = PropertyModel(input_size, n_diff_step, time_embed_dim).to(device)
sas_model = PropertyModel(input_size, n_diff_step, time_embed_dim).to(device)

molecule_dim = config["mol_model"]["d_model"] * config["mol_model"]["max_seq_length"]
protein_dim = config["protein_model"]["protein_embedding_dim"]
hidden_dim = config["pIC50_model"]["hidden_dim"]
num_heads = config["pIC50_model"]["num_heads"]
lr = config["pIC50_model"]["lr"]
num_epochs = config["pIC50_model"]["num_epochs"]
time_embed_dim = config["pIC50_model"]["time_embed_dim"]
pic50_model = pIC50Predictor(molecule_dim, protein_dim, num_heads, time_embed_dim, n_diff_step).to(device)

pic50_model.load_state_dict(torch.load('models/pIC50_model.pt', map_location=device))
qed_model.load_state_dict(torch.load('models/qed_model.pt', map_location=device))
sas_model.load_state_dict(torch.load('models/sas_model.pt', map_location=device))

diffusion_model = GaussianDiffusion(betas=get_named_beta_schedule(n_diff_step))
unet = Text2ImUNet(text_ctx=1, xf_width=protein_embedding_dim, xf_layers=0, xf_heads=0, xf_final_ln=0, tokenizer=None, in_channels=1, model_channels=48, out_channels=2, num_res_blocks=2, attention_resolutions=[], dropout=.1, channel_mult=(1, 2, 4, 8), dims=1)
unet.to(device)
unet.load_state_dict(torch.load('unet.pt', map_location=device))

if False:
    protein_sequence = "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQCSGVTFQ"
    protein_model_name = "facebook/esm2_t30_150M_UR50D"
    protein_tokenizer = EsmTokenizer.from_pretrained(protein_model_name)
    protein_model = EsmModel.from_pretrained(protein_model_name).to('cuda')
    encoded_protein = protein_tokenizer(protein_sequence, return_tensors='pt', padding=True, truncation=True).to('cuda')
    # Generate protein embeddings
    with torch.no_grad():
        protein_outputs = protein_model(**encoded_protein)
        protein_embeddings = protein_outputs.last_hidden_state

        # Mean and Max Pooling
        mean_pooled = protein_embeddings.mean(dim=1)
        max_pooled = protein_embeddings.max(dim=1).values
        combined_pooled = torch.cat((mean_pooled, max_pooled), dim=1)
    protein_embedding = combined_pooled.detach().to(device)
    np.save("data/3cl.npy", protein_embedding.detach().cpu().numpy())
else:
    protein_embedding = torch.tensor(np.load("data/3cl.npy"), device=device)

print(protein_embedding.shape)

# cond_fn = get_balanced_grad()

# sample = diffusion_model.p_sample_loop(unet, (1, 1, input_size), prot=protein_embedding, cond_fn=get_balanced_grad)
sample = diffusion_model.p_sample_loop(unet, (1, 1, input_size), prot=protein_embedding).reshape(input_size)

# sample = diffusion_model.ddim_sample_loop(unet, (1, 1, input_size), prot=protein_embedding)

mins = np.load("data/smiles_mins_selfies.npy")
maxes = np.load("data/smiles_maxes_selfies.npy")
sample_rescale = torch.tensor(((sample.cpu().detach().numpy() + 1) / 2) * (maxes - mins) + mins, device=device)
# print(sample_rescale.shape)
# print(maxes.shape)
# print(mins.shape)

tokenizer = SelfiesTok.load("models/selfies_tok.json")

with torch.no_grad():
    decoded_smiles, _ = mol_model.decode_representation(sample_rescale.reshape(1, max_seq_length, d_model), None, max_length=128, tokenizer=tokenizer)


predicted_selfies = tokenizer.decode(decoded_smiles.detach().cpu().flatten().tolist(), skip_special_tokens=True)
predicted_smiles = sf.decoder(predicted_selfies)
print("Sampled SELFIES:", predicted_selfies)
print("Sample SMILES: ", predicted_smiles)

try:
    mol = Chem.MolFromSmiles(predicted_smiles)
    img = Draw.MolToImage(mol)
    img_path = f'mol.png' 
    img.save(img_path)
except:
    pass


# final_x = sample_rescale.detach().to(device).squeeze(0)
# pic50_model.eval()
# qed_model.eval()
# sas_model.eval()
# pic50 = (pic50_model(final_x, protein_embedding, torch.tensor([1000], device=device))).detach().cpu()
# qed = (qed_model(final_x, torch.tensor([1000], device=device)).clamp(0, 1)).detach().cpu()
# sas = (sas_model(final_x, torch.tensor([1000], device=device)).clamp(1, 10)).detach().cpu()
# print(pic50)
# print(qed)
# print(sas)
