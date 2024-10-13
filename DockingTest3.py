from rdkit import Chem
from rdkit.Chem import AllChem
import os

# Generate 1000 molecules
# IMPORTS
import torch
import yaml
from DiffusionModelGLIDE3 import *
from tqdm import tqdm
from unet_condition3 import Text2ImUNet
from transformers import EsmTokenizer, EsmModel
from MolTransformerSelfies import MultiTaskTransformer
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from fcd_torch import FCD
from SelfiesTok import SelfiesTok
import numpy as np
import pandas as pd
import random
import umap
import matplotlib.pyplot as plt
from typing import Set, Any
from TestingUtils import *
from scipy.stats import wasserstein_distance
import sys
import os
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'share', 'RDKit', 'Contrib', 'SA_Score'))
import sascorer
from rdkit.Chem import Descriptors, QED
from collections import Counter
import selfies as sf
from pIC50Predictor2 import pIC50Predictor
import argparse

parser = argparse.ArgumentParser(description="Generate molecules with specified protein and parameters.")
# parser.add_argument("--protein", type=str, default="MASWSHPQFEKGGGARGGSGGGSWSHPQFEKGFDYKDDDDKGTMTEGARAADEVRVPLGAPPPGPAALVGASPESPGAPGREAERGSELGVSPSESPAAERGAELGADEEQRVPYPALAATVFFCLGQTTRPRSWCLRLVCNPWFEHVSMLVIMLNCVTLGMFRPCEDVECGSERCNILEAFDAFIFAFFAVEMVIKMVALGLFGQKCYLGDTWNRLDFFIVVAGMMEYSLDGHNVSLSAIRTVRVLRPLRAINRVPSMRILVTLLLDTLPMLGNVLLLCFFVFFIFGIVGVQLWAGLLRNRCFLDSAFVRNNNLTFLRPYYQTEEGEENPFICSSRRDNGMQKCSHIPGRRELRMPCTLGWEAYTQPQAEGVGAARNACINWNQYYNVCRSGDSNPHNGAINFDNIGYAWIAIFQVITLEGWVDIMYYVMDAHSFYNFIYFILLIIVGSFFMINLCLVVIATQFSETKQRESQLMREQRARHLSNDSTLASFSEPGSCYEELLKYVGHIFRKVKRRSLRLYARWQSRWRKKVDPGWMGRLWVTFSGKLRRIVDSKYFSRGIMMAILVNTLSMGVEYHEQPEELTNALEISNIVFTSMFALEMLLKLLACGPLGYIRNPYNIFDGIIVVISVWEIVGQADGGLSVLRTFRLLRVLKLVRFLPALRRQLVVLVKTMDNVATFCTLLMLFIFIFSILGMHLFGCKFSLKTDTGDTVPDRKNFDSLLWAIVTVFQILTQEDWNVVLYNGMASTSSWAALYFVALMTFGNYVLFNLLVAILVEGFQAEGDANRSDTDEDKTSVHFEEDFHKLRELQTTELKMCSLAVTPNGHLEGRGSLSPPLIMCTAATPMPTPKSSPFLDAAPSLPDSRRGSSSSGDPPLGDQKPPASLRSSPCAPWGPSGAWSSRRSSWSSLGRAPSLKRRGQCGERESLLSGEGKGSTDDEAEDGRAAPGPRATPLRRAESLDPRPLRPAALPPTKCRDRDGQVVALPSDFFLRIDSHREDAAELDDDSEDSCCLRLHKVLEPYKPQWCRSREAWALYLFSPQNRFRVSCQKVITHKMFDHVVLVFIFLNCVTIALERPDIDPGSTERVFLSVSNYIFTAIFVAEMMVKVVALGLLSGEHAYLQSSWNLLDGLLVLVSLVDIVVAMASAGGAKILGVLRVLRLLRTLRPLRVISRAPGLKLVVETLISSLRPIGNIVLICCAFFIIFGILGVQLFKGKFYYCEGPDTRNISTKAQCRAAHYRWVRRKYNFDNLGQALMSLFVLSSKDGWVNIMYDGLDAVGVDQQPVQNHNPWMLLYFISFLLIVSFFVLNMFVGVVVENFHKCRQHQEAEEARRREEKRLRRLERRRRSTFPSPEAQRRPYYADYSPTRRSIHSLCTSHYLDLFITFIICVNVITMSMEHYNQPKSLDEALKYCNYVFTIVFVFEAALKLVAFGFRRFFKDRWNQLDLAIVLLSLMGITLEEIEMSAALPINPTIIRIMRVLRIARVLKLLKMATGMRALLDTVVQALPQVGNLGLLFMLLFFIYAALGVELFGRLECSEDNPCEGLSRHATFSNFGMAFLTLFRVSTGDNWNGIMKDTLRECSREDKHCLSYLPALSPVYFVTFVLVAQFVLVNVVVAVLMKHLEESNKEAREDAELDAEIELEMAQGPGSARRVDADRPPLPQESPGARDAPNLVARKVSVSRMLSLPNDSYMFRPVVPASAPHPRPLQEVEMETYGAGTPLGSVASVHSPPAESCASLQIPLAVSSPARSGEPLHALSPRGTARSPSLSRLLCRQEAVHTDSLEGKIDSPRDTLDPAEPGEKTPVRPVTQGGSLQSPPRSPRPASVRTRKHTFGQRCVSSRPAAPGGEEAEASDPADEEVSHITSSACPWQPTAEPHGPEASPVAGGERDLRRLYSVDAQGFLDKPGRADEQWRPSAELGSGEPGEAKAWGPEAEPALGARRKKKMSPPCISVEPPAEDEGSARPSAAEGGSTTLRRRTPSCEATPHRDSLEPTEGSGAGGDPAAKGERWGQASCRAEHLTVPSFAFEPLDLGVPSGDPFLDGSHSVTPESRASSSGAIVPLEPPESEPPMPVGDPPEKRRGLYLTVPQCPLEKPGSPSATPAPGGGADDPV", help="Protein sequence to be used")
parser.add_argument("--w", type=float, default=50, help="Weighting parameter")
parser.add_argument("--num_proteins", type=int, default=100, help="Number of proteins to generate molecules for")
args = parser.parse_args()

# Step 0: Load and generate
# Step 0.1: Load in the training data (potential source of issue as this includes val but whatever)
ref_file_smiles = "data/protein_drug_pairs_with_sequences_and_smiles.csv"
ref_file_finger = "data/smiles_output_selfies.npy"
ref_data_finger = np.load(ref_file_finger)
ref_data_smiles = pd.read_csv(ref_file_smiles)

# Step 0.2: Load in GaussianDiffusion model and unet model and MolTransformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

n_diff_step = config["diffusion_model"]["num_diffusion_steps"]
protein_embedding_dim = config["protein_model"]["protein_embedding_dim"]
batch_size = config["diffusion_model"]["batch_size"]
src_vocab_size = config["mol_model"]["src_vocab_size"]
tgt_vocab_size = config["mol_model"]["tgt_vocab_size"]
max_seq_length = config["mol_model"]["max_seq_length"]
num_tasks = config["mol_model"]["num_tasks"]
d_model = config["mol_model"]["d_model"]
num_heads = config["mol_model"]["num_heads"]
num_layers = config["mol_model"]["num_layers"]
d_ff = config["mol_model"]["d_ff"]
dropout = config["mol_model"]["dropout"]

diffusion_model = GaussianDiffusion(betas=get_named_beta_schedule(n_diff_step))
unet = Text2ImUNet(text_ctx=1, xf_width=protein_embedding_dim, xf_layers=0, xf_heads=0, xf_final_ln=0, tokenizer=None, in_channels=256, model_channels=256, out_channels=512, num_res_blocks=2, attention_resolutions=[], dropout=.1, channel_mult=(1, 2, 4, 8), dims=1)
mol_model = MultiTaskTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks).to(device)

# unet.load_state_dict(torch.load('unet_resized_even_attention_4_tuned.pt', map_location=device))
checkpoint = torch.load('checkpoint.pt', map_location=device)
unet.load_state_dict(checkpoint["state_dict"])

# exit()

mol_model.load_state_dict(torch.load('models/selfies_transformer_final.pt', map_location=device))

unet, mol_model = unet.to(device), mol_model.to(device)

# Step 0.3: Pick 40 random proteins and generate 25 molecules per protein
# Group by Protein Sequence and filter groups that have at least 25 SMILES strings
ref_data_smiles.reset_index(inplace=True)
# Get ESM
# protein_fingers_full = np.load("data/protein_embeddings.npy")
# protein_finger = protein_fingers_full[123]
# protein_finger = np.load("data/3cl.npy").reshape(1, -1)

# print(len(protein_fingers_cut))

# pic50_model = pIC50Predictor(n_diff_step, 1280).to(device)
# pic50_model.load_state_dict(torch.load('models/pIC50_model.pt', map_location=device))

# def get_pIC50_grad(x, prot, t):
#     pic50_model.train()
#     x = x.detach().requires_grad_(True)
#     # print(x)
#     # print(prot)
#     # print(t)
#     # print(pic50_model.training)
#     pic50_pred = pic50_model(x, prot, t)
#     # gradients = []
#     # for i in range(x.shape[0]):
#     #     grad = torch.autograd.grad(outputs=pic50_pred[i], inputs=x[i])[0].deatch()
#     #     gradients.append(grad)
#     # grad = torch.stack(gradients)
#     # print(pic50_pred.requires_grad)
#     grad = torch.autograd.grad(pic50_pred.sum(), x)[0].detach()
#     return grad

# Step 0.3.1: Generate 25 molecules per protein

# protein_sequence = "MASWSHPQFEKGGGARGGSGGGSWSHPQFEKGFDYKDDDDKGTMTEGARAADEVRVPLGAPPPGPAALVGASPESPGAPGREAERGSELGVSPSESPAAERGAELGADEEQRVPYPALAATVFFCLGQTTRPRSWCLRLVCNPWFEHVSMLVIMLNCVTLGMFRPCEDVECGSERCNILEAFDAFIFAFFAVEMVIKMVALGLFGQKCYLGDTWNRLDFFIVVAGMMEYSLDGHNVSLSAIRTVRVLRPLRAINRVPSMRILVTLLLDTLPMLGNVLLLCFFVFFIFGIVGVQLWAGLLRNRCFLDSAFVRNNNLTFLRPYYQTEEGEENPFICSSRRDNGMQKCSHIPGRRELRMPCTLGWEAYTQPQAEGVGAARNACINWNQYYNVCRSGDSNPHNGAINFDNIGYAWIAIFQVITLEGWVDIMYYVMDAHSFYNFIYFILLIIVGSFFMINLCLVVIATQFSETKQRESQLMREQRARHLSNDSTLASFSEPGSCYEELLKYVGHIFRKVKRRSLRLYARWQSRWRKKVDPGWMGRLWVTFSGKLRRIVDSKYFSRGIMMAILVNTLSMGVEYHEQPEELTNALEISNIVFTSMFALEMLLKLLACGPLGYIRNPYNIFDGIIVVISVWEIVGQADGGLSVLRTFRLLRVLKLVRFLPALRRQLVVLVKTMDNVATFCTLLMLFIFIFSILGMHLFGCKFSLKTDTGDTVPDRKNFDSLLWAIVTVFQILTQEDWNVVLYNGMASTSSWAALYFVALMTFGNYVLFNLLVAILVEGFQAEGDANRSDTDEDKTSVHFEEDFHKLRELQTTELKMCSLAVTPNGHLEGRGSLSPPLIMCTAATPMPTPKSSPFLDAAPSLPDSRRGSSSSGDPPLGDQKPPASLRSSPCAPWGPSGAWSSRRSSWSSLGRAPSLKRRGQCGERESLLSGEGKGSTDDEAEDGRAAPGPRATPLRRAESLDPRPLRPAALPPTKCRDRDGQVVALPSDFFLRIDSHREDAAELDDDSEDSCCLRLHKVLEPYKPQWCRSREAWALYLFSPQNRFRVSCQKVITHKMFDHVVLVFIFLNCVTIALERPDIDPGSTERVFLSVSNYIFTAIFVAEMMVKVVALGLLSGEHAYLQSSWNLLDGLLVLVSLVDIVVAMASAGGAKILGVLRVLRLLRTLRPLRVISRAPGLKLVVETLISSLRPIGNIVLICCAFFIIFGILGVQLFKGKFYYCEGPDTRNISTKAQCRAAHYRWVRRKYNFDNLGQALMSLFVLSSKDGWVNIMYDGLDAVGVDQQPVQNHNPWMLLYFISFLLIVSFFVLNMFVGVVVENFHKCRQHQEAEEARRREEKRLRRLERRRRSTFPSPEAQRRPYYADYSPTRRSIHSLCTSHYLDLFITFIICVNVITMSMEHYNQPKSLDEALKYCNYVFTIVFVFEAALKLVAFGFRRFFKDRWNQLDLAIVLLSLMGITLEEIEMSAALPINPTIIRIMRVLRIARVLKLLKMATGMRALLDTVVQALPQVGNLGLLFMLLFFIYAALGVELFGRLECSEDNPCEGLSRHATFSNFGMAFLTLFRVSTGDNWNGIMKDTLRECSREDKHCLSYLPALSPVYFVTFVLVAQFVLVNVVVAVLMKHLEESNKEAREDAELDAEIELEMAQGPGSARRVDADRPPLPQESPGARDAPNLVARKVSVSRMLSLPNDSYMFRPVVPASAPHPRPLQEVEMETYGAGTPLGSVASVHSPPAESCASLQIPLAVSSPARSGEPLHALSPRGTARSPSLSRLLCRQEAVHTDSLEGKIDSPRDTLDPAEPGEKTPVRPVTQGGSLQSPPRSPRPASVRTRKHTFGQRCVSSRPAAPGGEEAEASDPADEEVSHITSSACPWQPTAEPHGPEASPVAGGERDLRRLYSVDAQGFLDKPGRADEQWRPSAELGSGEPGEAKAWGPEAEPALGARRKKKMSPPCISVEPPAEDEGSARPSAAEGGSTTLRRRTPSCEATPHRDSLEPTEGSGAGGDPAAKGERWGQASCRAEHLTVPSFAFEPLDLGVPSGDPFLDGSHSVTPESRASSSGAIVPLEPPESEPPMPVGDPPEKRRGLYLTVPQCPLEKPGSPSATPAPGGGADDPV"

protein_sequence = "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQCSGVTFQ"
# protein_model_name = "facebook/esm2_t30_150M_UR50D"
# protein_tokenizer = EsmTokenizer.from_pretrained(protein_model_name)
# protein_model = EsmModel.from_pretrained(protein_model_name).to('cuda')
# encoded_protein = protein_tokenizer(protein_sequence, return_tensors='pt', padding=True, truncation=True).to('cuda')
# Generate protein embeddings
# with torch.no_grad():
    # protein_outputs = protein_model(**encoded_protein)
    # protein_embeddings = protein_outputs.last_hidden_state

    # Mean and Max Pooling
    # mean_pooled = protein_embeddings.mean(dim=1)
    # max_pooled = protein_embeddings.max(dim=1).values
    # combined_pooled = torch.cat((mean_pooled, max_pooled), dim=1)
# protein_embedding = combined_pooled.detach().cpu().numpy()
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
xf_in = tokenizer(protein_sequence, return_tensors="pt", padding=True, truncation=True, max_length=1026)
protein_ids = xf_in["input_ids"]
protein_atts = xf_in["attention_mask"]

protein_ids_repeat = protein_ids.repeat(args.num_proteins, 1).to("cuda")
protein_atts_repeat = protein_atts.repeat(args.num_proteins, 1).to("cuda")
# protein_finger = torch.tensor(protein_embedding.reshape(1, -1), device=device)
# protein_finger = protein_finger.repeat(args.num_proteins, 1)

# exit()

sample = diffusion_model.p_sample_loop(unet, (args.num_proteins, 256, 128), prot=protein_ids_repeat, attn=protein_atts_repeat, w=args.w).detach().reshape(args.num_proteins, 1, 32768)

# Step 0.3.2: Remember to unnormalize
mins = torch.tensor(np.load("data/smiles_mins_selfies.npy"), device=device).reshape(1, 1, -1)
maxes = torch.tensor(np.load("data/smiles_maxes_selfies.npy"), device=device).reshape(1, 1, -1)
sample_rescale = (((sample + 1) / 2) * (maxes - mins) + mins)
# print(sample_rescale.shape)

# Step 0.3.3: Convert from SELFIES back to SMILES
tokenizer = SelfiesTok.load("models/selfies_tok.json")
with torch.no_grad():
    decoded_smiles, _ = mol_model.decode_representation(sample_rescale.reshape(-1, max_seq_length, d_model), None, max_length=128, tokenizer=tokenizer)

gen_smiles = []
gen_smiles_filtered = []
qeds = []
sass = []

for decode in decoded_smiles:
    predicted_selfie = tokenizer.decode(decode.detach().cpu().flatten().tolist(), skip_special_tokens=True)
    predicted_smile = sf.decoder(predicted_selfie)
    
    try:
        mol = Chem.MolFromSmiles(predicted_smile)
        # mol = Chem.AddHs(mol)
        # Filter for QED at least 0.5, SAS below 5
        qed = QED.qed(mol)
        sas = sascorer.calculateScore(mol)
        if qed >= .4 and sas <= 6:
            # print(f"added qed: {qed} sas: {sas}")
            # qeds.append(qed)
            # sass.append(sas)
            gen_smiles_filtered.append(predicted_smile)
    except:
        pass

    gen_smiles.append(predicted_smile)

# print(sum(qeds)/len(qeds))
# print(sum(sass)/len(sass))
print(len(gen_smiles_filtered))

idx = 0
for smile in gen_smiles:
    # Convert SMILES to a molecule object
    if smile is not None and smile != "":
        mol = Chem.MolFromSmiles(smile)
        # Generate 3D coordinates (optional, if you want to have a 3D structure in the SDF)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        # Write the molecule to an SDF file
        with Chem.SDWriter(f'ligands_cl3/ligand{idx:03}.sdf') as writer:
            writer.write(mol)
        idx += 1

idx = 0
for smile in gen_smiles_filtered:
    if smile is not None and smile != "":
        # Convert SMILES to a molecule object
        mol = Chem.MolFromSmiles(smile)
        # Generate 3D coordinates (optional, if you want to have a 3D structure in the SDF)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        # Write the molecule to an SDF file
        with Chem.SDWriter(f'ligands_cl3_filtered/ligand{idx:03}.sdf') as writer:
            writer.write(mol)
        idx += 1

