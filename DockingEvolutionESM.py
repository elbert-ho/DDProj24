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
from rdkit.Chem import Draw, AllChem, Descriptors, QED
from rdkit.DataStructs import FingerprintSimilarity
from SelfiesTok import SelfiesTok
import numpy as np
import pandas as pd
import random
import umap
import matplotlib.pyplot as plt
from TestingUtils import *
import sys
import os
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'share', 'RDKit', 'Contrib', 'SA_Score'))
import sascorer
import selfies as sf
import argparse
from ESM2Regressor import ESM2Regressor


parser = argparse.ArgumentParser(description="Generate molecules with specified protein and parameters.")
args = parser.parse_args()

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

# unet.load_state_dict(torch.load('unet_resized_odd.pt', map_location=device))
mol_model.load_state_dict(torch.load('models/selfies_transformer_final.pt', map_location=device))

checkpoint = torch.load('checkpoint.pt', map_location=device)
unet.load_state_dict(checkpoint["state_dict"])

unet, mol_model = unet.to(device), mol_model.to(device)

# REFERENCE SAMPLE SHOULD BE UNNORMALIZED
# reference_sample = torch.tensor(np.load("data/ref_sample.npy"), device=device).reshape(1, 1, 32768)
reference_sample = torch.tensor(np.load("data/smiles_output_selfies_normal2.npy")[0], device=device).reshape(1, 1, 32768)

# reference_sample = torch.tensor(np.load("data/smiles_output_selfies_normal.npy")[9653], device=device).reshape(1, 1, 32768)

# Get Reference SMILES
mins = torch.tensor(np.load("data/smiles_mins_selfies.npy"), device=device).reshape(1, 1, -1)
maxes = torch.tensor(np.load("data/smiles_maxes_selfies.npy"), device=device).reshape(1, 1, -1)
reference_sample_rescale = (((reference_sample + 1) / 2) * (maxes - mins) + mins)
# print(sample_rescale.shape)

# Step 0.3.3: Convert from SELFIES back to SMILES
tokenizer = SelfiesTok.load("models/selfies_tok.json")
with torch.no_grad():
    decoded_smiles, _ = mol_model.decode_representation(reference_sample_rescale.reshape(-1, max_seq_length, d_model), None, max_length=128, tokenizer=tokenizer)

predicted_selfie = tokenizer.decode(decoded_smiles[0].detach().cpu().flatten().tolist(), skip_special_tokens=True)
reference_smile = sf.decoder(predicted_selfie)

reference_mol = Chem.MolFromSmiles(reference_smile)
reference_fp = AllChem.GetMorganFingerprintAsBitVect(reference_mol, radius=2, nBits=2048)

# model_reg = ESM2Regressor()
# model_reg.load_state_dict(torch.load('esm2_regressor_saved.pth', map_location="cuda"))
# model_reg.to("cuda")
# model_reg.eval()

# protein_sequence = "MASWSHPQFEKGGGARGGSGGGSWSHPQFEKGFDYKDDDDKGTMTEGARAADEVRVPLGAPPPGPAALVGASPESPGAPGREAERGSELGVSPSESPAAERGAELGADEEQRVPYPALAATVFFCLGQTTRPRSWCLRLVCNPWFEHVSMLVIMLNCVTLGMFRPCEDVECGSERCNILEAFDAFIFAFFAVEMVIKMVALGLFGQKCYLGDTWNRLDFFIVVAGMMEYSLDGHNVSLSAIRTVRVLRPLRAINRVPSMRILVTLLLDTLPMLGNVLLLCFFVFFIFGIVGVQLWAGLLRNRCFLDSAFVRNNNLTFLRPYYQTEEGEENPFICSSRRDNGMQKCSHIPGRRELRMPCTLGWEAYTQPQAEGVGAARNACINWNQYYNVCRSGDSNPHNGAINFDNIGYAWIAIFQVITLEGWVDIMYYVMDAHSFYNFIYFILLIIVGSFFMINLCLVVIATQFSETKQRESQLMREQRARHLSNDSTLASFSEPGSCYEELLKYVGHIFRKVKRRSLRLYARWQSRWRKKVDPGWMGRLWVTFSGKLRRIVDSKYFSRGIMMAILVNTLSMGVEYHEQPEELTNALEISNIVFTSMFALEMLLKLLACGPLGYIRNPYNIFDGIIVVISVWEIVGQADGGLSVLRTFRLLRVLKLVRFLPALRRQLVVLVKTMDNVATFCTLLMLFIFIFSILGMHLFGCKFSLKTDTGDTVPDRKNFDSLLWAIVTVFQILTQEDWNVVLYNGMASTSSWAALYFVALMTFGNYVLFNLLVAILVEGFQAEGDANRSDTDEDKTSVHFEEDFHKLRELQTTELKMCSLAVTPNGHLEGRGSLSPPLIMCTAATPMPTPKSSPFLDAAPSLPDSRRGSSSSGDPPLGDQKPPASLRSSPCAPWGPSGAWSSRRSSWSSLGRAPSLKRRGQCGERESLLSGEGKGSTDDEAEDGRAAPGPRATPLRRAESLDPRPLRPAALPPTKCRDRDGQVVALPSDFFLRIDSHREDAAELDDDSEDSCCLRLHKVLEPYKPQWCRSREAWALYLFSPQNRFRVSCQKVITHKMFDHVVLVFIFLNCVTIALERPDIDPGSTERVFLSVSNYIFTAIFVAEMMVKVVALGLLSGEHAYLQSSWNLLDGLLVLVSLVDIVVAMASAGGAKILGVLRVLRLLRTLRPLRVISRAPGLKLVVETLISSLRPIGNIVLICCAFFIIFGILGVQLFKGKFYYCEGPDTRNISTKAQCRAAHYRWVRRKYNFDNLGQALMSLFVLSSKDGWVNIMYDGLDAVGVDQQPVQNHNPWMLLYFISFLLIVSFFVLNMFVGVVVENFHKCRQHQEAEEARRREEKRLRRLERRRRSTFPSPEAQRRPYYADYSPTRRSIHSLCTSHYLDLFITFIICVNVITMSMEHYNQPKSLDEALKYCNYVFTIVFVFEAALKLVAFGFRRFFKDRWNQLDLAIVLLSLMGITLEEIEMSAALPINPTIIRIMRVLRIARVLKLLKMATGMRALLDTVVQALPQVGNLGLLFMLLFFIYAALGVELFGRLECSEDNPCEGLSRHATFSNFGMAFLTLFRVSTGDNWNGIMKDTLRECSREDKHCLSYLPALSPVYFVTFVLVAQFVLVNVVVAVLMKHLEESNKEAREDAELDAEIELEMAQGPGSARRVDADRPPLPQESPGARDAPNLVARKVSVSRMLSLPNDSYMFRPVVPASAPHPRPLQEVEMETYGAGTPLGSVASVHSPPAESCASLQIPLAVSSPARSGEPLHALSPRGTARSPSLSRLLCRQEAVHTDSLEGKIDSPRDTLDPAEPGEKTPVRPVTQGGSLQSPPRSPRPASVRTRKHTFGQRCVSSRPAAPGGEEAEASDPADEEVSHITSSACPWQPTAEPHGPEASPVAGGERDLRRLYSVDAQGFLDKPGRADEQWRPSAELGSGEPGEAKAWGPEAEPALGARRKKKMSPPCISVEPPAEDEGSARPSAAEGGSTTLRRRTPSCEATPHRDSLEPTEGSGAGGDPAAKGERWGQASCRAEHLTVPSFAFEPLDLGVPSGDPFLDGSHSVTPESRASSSGAIVPLEPPESEPPMPVGDPPEKRRGLYLTVPQCPLEKPGSPSATPAPGGGADDPV"
# protein_sequence = "MARKKLKKFTTLEIVLSVLLLVLFIISIVLIVLLAKESLKSTAPDPGTTGTPDPGTTGTPDPGTTGTTHARTTGPPDPGTTGTTPVSAECPVVNELERINCIPDQPPTKATCDQRGCCWNPQGAVSVPWCYYSKNHSYHVEGNLVNTNAGFTARLKNLPSSPVFGSNVDNVLLTAEYQTSNRFHFKLTDQTNNRFEVPHEHVQSFSGNAAASLTYQVEISRQPFSIKVTRRSNNRVLFDSSIGPLLFADQFLQLSTRLPSTNVYGLGEHVHQQYRHDMNWKTWPIFNRDTTPNGNGTNLYGAQTFFLCLEDASGLSFGVFLMNSNAMEVVLQPAPAITYRTIGGILDFYVFLGNTPEQVVQEYLELIGRPALPSYWALGFHLSRYEYGTLDNMREVVERNRAAQLPYDVQHADIDYMDERRDFTYDSVDFKGFPEFVNELHNNGQKLVIIVDPAISNNSSSSKPYGPYDRGSDMKIWVNSSDGVTPLIGEVWPGQTVFPDYTNPNCAVWWTKEFELFHNQVEFDGIWIDMNEVSNFVDGSVSGCSTNNLNNPPFTPRILDGYLFCKTLCMDAVQHWGKQYDIHNLYGYSMAVATAEAAKTVFPNKRSFILTRSTFAGSGKFAAHWLGDNTATWDDLRWSIPGVLEFNLFGIPMVGPDICGFALDTPEELCRRWMQLGAFYPFSRNHNGQGYKDQDPASFGADSLLLNSSRHYLNIRYTLLPYLYTLFFRAHSRGDTVARPLLHEFYEDNSTWDVHQQFLWGPGLLITPVLDEGAEKVMAYVPDAVWYDYETGSQVRWRKQKVEMELPGDKIGLHLRGGYIFPTQQPNTTTLASRKNPLGLIIALDENKEAKGELFWDNGETKDTVANKVYLLCEFSVTQNRLEVNISQSTYKDPNNLAFNEIKILGTEEPSNVTVKHNGVPSQTSPTVTYDSNLKVAIITDIDLLLGEAYTVEWSIKIRDEEKIDCYPDENGASAENCTARGCIWEASNSSGVPFCYFVNDLYSVSDVQYNSHGATADISLKSSVYANAFPSTPVNPLRLDVTYHKNEMLQFKIYDPNKNRYEVPVPLNIPSMPSSTPEGQLYDVLIKKNPFGIEIRRKSTGTIIWDSQLLGFTFSDMFIRISTRLPSKYLYGFGETEHRSYRRDLEWHTWGMFSRDQPPGYKKNSYGVHPYYMGLEEDGSAHGVLLLNSNAMDVTFQPLPALTYRTTGGVLDFYVFLGPTPELVTQQYTELIGRPVMVPYWSLGFQLCRYGYQNDSEIASLYDEMVAAQIPYDVQYSDIDYMERQLDFTLSPKFAGFPALINRMKADGMRVILILDPAISGNETQPYPAFTRGVEDDVFIKYPNDGDIVWGKVWPDFPDVVVNGSLDWDSQVELYRAYVAFPDFFRNSTAKWWKREIEELYNNPQNPERSLKFDGMWIDMNEPSSFVNGAVSPGCRDASLNHPPYMPHLESRDRGLSSKTLCMESQQILPDGSLVQHYNVHNLYGWSQTRPTYEAVQEVTGQRGVVITRSTFPSSGRWAGHWLGDNTAAWDQLKKSIIGMMEFSLFGISYTGADICGFFQDAEYEMCVRWMQLGAFYPFSRNHNTIGTRRQDPVSWDAAFVNISRNVLQTRYTLLPYLYTLMQKAHTEGVTVVRPLLHEFVSDQVTWDIDSQFLLGPAFLVSPVLERNARNVTAYFPRARWYDYYTGVDINARGEWKTLPAPLDHINLHVRGGYILPWQEPALNTHLSRKNPLGLIIALDENKEAKGELFWDDGQTKDTVAKKVYLLCEFSVTQNHLEVTISQSTYKDPNNLAFNEIKILGMEEPSNVTVKHNGVPSQTSPTVTYDSNLKVAIITDINLFLGEAYTVEWSIKIRDEEKIDCYPDENGDSAENCTARGCIWEASNSSGVPFCYFVNDLYSVSDVQYNSHGATADISLKSSVHANAFPSTPVNPLRLDVTYHKNEMLQFKIYDPNNNRYEVPVPLNIPSVPSSTPEGQLYDVLIKKNPFGIEIRRKSTGTIIWDSQLLGFTFNDMFIRISTRLPSKYLYGFGETEHTSYRRDLEWHTWGMFSRDQPPGYKKNSYGVHPYYMGLEEDGSAHGVLLLNSNAMDVTFQPLPALTYRTTGGVLDFYVFLGPTPELVTQQYTELIGRPVMVPYWSLGFQLCRYGYQNDSEISSLYDEMVAAQIPYDVQYSDIDYMERQLDFTLSPKFAGFPALINRMKADGMRVILILDPAISGNETQPYPAFTRGVEDDVFIKYPNDGDIVWGKVWPDFPDVVVNGSLDWDSQVELYRAYVAFPDFFRNSTAKWWKREIEELYNNPQNPERSLKFDGMWIDMNEPSSFVNGAVSPGCRDASLNHPPYMPYLESRDRGLSSKTLCMESQQILPDGSPVQHYNVHNLYGWSQTRPTYEAVQEVTGQRGVVITRSTFPSSGRWAGHWLGDNTAAWDQLKKSIIGMMEFSLFGISYTGADICGFFQDAEYEMCVRWMQLGAFYPFSRNHNTIGTRRQDPVSWDVAFVNISRTVLQTRYTLLPYLYTLMHKAHTEGVTVVRPLLHEFVSDQVTWDIDSQFLLGPAFLVSPVLERNARNVTAYFPRARWYDYYTGVDINARGEWKTLPAPLDHINLHVRGGYILPWQEPALNTHLSRQKFMGFKIALDDEGTAGGWLFWDDGQSIDTYGKGLYYLASFSASQNTMQSHIIFNNYITGTNPLKLGYIEIWGVGSVPVTSVSISVSGMVITPSFNNDPTTQVLSIDVTDRNISLHNFTSLTWISTL"

# protein_sequence = "MSGPRAGFYRQELNKTVWEVPQRLQGLRPVGSGAYGSVCSAYDARLRQKVAVKKLSRPFQSLIHARRTYRELRLLKHLKHENVIGLLDVFTPATSIEDFSEVYLVTTLMGADLNNIVKCQALSDEHVQFLVYQLLRGLKYIHSAGIIHRDLKPSNVAVNEDCELRILDFGLARQADEEMTGYVATRWYRAPEIMLNWMHYNQTVDIWSVGCIMAELLQGKALFPGSDYIDQLKRIMEVVGTPSPEVLAKISSEHARTYIQSLPPMPQKDLSSIFRGANPLAIDLLGRMLVLDSDQRVSAAEALAHAYFSQYHDPEDEPEAEPYDESVEAKERTLEEWKELTYQEVLSFKPPEPPKPPGSLEIEQ"

# protein_model_name = "facebook/esm2_t6_8M_UR50D"
# protein_tokenizer = EsmTokenizer.from_pretrained(protein_model_name)
# protein_model = EsmModel.from_pretrained(protein_model_name).to('cuda')
# encoded_protein = protein_tokenizer(protein_sequence, return_tensors='pt', padding=True, truncation=True).to('cuda')
# # Generate protein embeddings
# with torch.no_grad():
#     protein_outputs = protein_model(**encoded_protein)
#     protein_embeddings = protein_outputs.last_hidden_state

#     # representation = model_reg.get_rep(protein_sequence).flatten().detach().cpu().numpy()
#     # Mean and Max Pooling
#     mean_pooled = protein_embeddings.mean(dim=1)
#     # max_pooled = protein_embeddings.max(dim=1).values
#     # combined_pooled = torch.cat((mean_pooled, max_pooled), dim=1)
#     combined_pooled = mean_pooled
# # protein_embedding = representation
# protein_embedding = combined_pooled

# protein_finger1 = torch.tensor(protein_embedding.reshape(1, -1), device=device)
# protein_finger = protein_finger1.repeat(50, 1)

# protein_finger = (protein_sequence,) * 50

protein_finger = None

reference_sample_flattened = reference_sample.reshape(1, -1)
reference_sample = reference_sample.reshape(-1, 256, 128).repeat(50, 1, 1)

qed_lists = []
sas_lists = []
sim_lists = []

# print(reference_sample)

# for time_test in range(1):
#     dict = diffusion_model.training_losses(unet, reference_sample, torch.tensor([time_test], device=device).repeat(50), protein_finger)
#     print(dict["loss"].mean())
#     print(dict["vb"].mean())
#     print(dict["mse"].mean())
# exit()

protein_ids = torch.tensor(np.load("data/input_ids2.npy")[0], dtype=torch.int32).reshape(1, -1).to(device)
protein_atts = torch.tensor(np.load("data/attention_masks2.npy")[0], dtype=torch.int32).reshape(1, -1).to(device)

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

# df = pd.read_csv("data/protein_drug_pairs_with_sequences_and_smiles_cleaned2.csv")
# protein_strings = df["Protein Sequence"].to_list()

# protein_strings = "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQCSGVTFQ"
# xf_in = tokenizer(protein_strings, return_tensors="pt", padding=True, truncation=True, max_length=1026)
# protein_ids = xf_in["input_ids"]
# protein_atts = xf_in["attention_mask"]

protein_ids_repeat = protein_ids.repeat(50, 1).to("cuda")
protein_atts_repeat = protein_atts.repeat(50, 1).to("cuda")

if True:
    # ws = [10, 20, 50]
    # for w_test in ws:
        # print(w_test)
        # step = 999
    for step in range(999, 1000, 200):
    # step = 999
    # for step in range(10, 101, 10):
    # for step in range(1, 11, 1):
        time = torch.tensor([step], device=device).repeat(50, 1)

        img = diffusion_model.q_sample(reference_sample, time)

        # print(reference_sample[0][0])
        # print(img[0][0])

        shape = (50, 256, 128)
        indices = list(range(step))[::-1]
        # Lazy import so that we don't depend on tqdm.
        from tqdm.auto import tqdm
        indices = tqdm(indices)
        for i in indices:
            # print((reference_sample - img).pow(2).sum().sqrt())
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = diffusion_model.p_sample(
                    unet,
                    img,
                    t,
                    prot=protein_ids_repeat,
                    attn=protein_atts_repeat,
                    w=5,
                    clip_denoised=True,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=None,
                )
                img = out["sample"]

        # print(img[0][0])

        # print(torch.sqrt((reference_sample[0][0] - img[0][0]).pow(2).sum() / torch.numel(img[0][0])))

        sample = img.reshape(-1, 1, 32768)

        # for a_sample in sample:
            # print((a_sample - reference_sample_flattened).pow(2).sum().sqrt())

        mins = torch.tensor(np.load("data/smiles_mins_selfies.npy"), device=device).reshape(1, 1, -1)
        maxes = torch.tensor(np.load("data/smiles_maxes_selfies.npy"), device=device).reshape(1, 1, -1)
        sample_rescale = (((sample + 1) / 2) * (maxes - mins) + mins)
        # print(sample_rescale.shape)

        # Step 0.3.3: Convert from SELFIES back to SMILES
        tokenizer = SelfiesTok.load("models/selfies_tok.json")
        with torch.no_grad():
            decoded_smiles, _ = mol_model.decode_representation(sample_rescale.reshape(-1, max_seq_length, d_model), None, max_length=128, tokenizer=tokenizer)

        qeds = []
        sass = []
        sims = []
        final_smiles = []
        for decode in decoded_smiles:
            predicted_selfie = tokenizer.decode(decode.detach().cpu().flatten().tolist(), skip_special_tokens=True)
            predicted_smile = sf.decoder(predicted_selfie)
            
            try:
                mol = Chem.MolFromSmiles(predicted_smile)
                # mol = Chem.AddHs(mol)
                # Filter for QED at least 0.5, SAS below 5
                qed = QED.qed(mol)
                sas = sascorer.calculateScore(mol)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                
                final_smiles.append(predicted_smile)
                qeds.append(qed)
                sass.append(sas)
                sims.append(FingerprintSimilarity(reference_fp, fp))
            except:
                pass

        print(np.mean(calculate_internal_pairwise_similarities(final_smiles)))
        qed_lists.append(qeds)
        sas_lists.append(sass)
        sim_lists.append(sims)

    reference_qed = QED.qed(reference_mol)
    reference_sas = sascorer.calculateScore(reference_mol)

    import seaborn as sns
    x_labels = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=qed_lists)

    # Customize the x-axis labels
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45)
    plt.axhline(y=reference_qed, color='orange', linestyle='--', label='Reference')

    # Add titles and labels
    plt.title('QED scores vs. Time Step')
    plt.xlabel('Time Step')
    plt.ylabel('QED score')

    # Add a legend
    plt.legend()
    # Show the plot
    plt.savefig("images2/evoQED.png")
    plt.clf()

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=sas_lists)

    # Customize the x-axis labels
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45)
    plt.axhline(y=reference_sas, color='orange', linestyle='--', label='Reference')

    # Add titles and labels
    plt.title('SA scores vs. Time Step')
    plt.xlabel('Time Step')
    plt.ylabel('SA score')

    # Add a legend
    plt.legend()
    # Show the plot
    plt.savefig("images2/evoSAS.png")
    plt.clf()

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=sim_lists)

    # Customize the x-axis labels
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45)
    # Add titles and labels
    plt.title('Sims vs. Time Step')
    plt.xlabel('Time Step')
    plt.ylabel('Sims')

    # Add a legend
    plt.legend()
    # Show the plot
    plt.savefig("images2/evoSim.png")
    exit()

num_advance = 150

per_new = 100

time = torch.tensor([num_advance], device=device).repeat(50, 1)
# Generation 0
img = diffusion_model.q_sample(reference_sample, time)
shape = (50, 256, 128)
indices = list(range(num_advance))[::-1]
# Lazy import so that we don't depend on tqdm.
from tqdm.auto import tqdm
indices = tqdm(indices)
for i in indices:
    t = torch.tensor([i] * shape[0], device=device)
    with torch.no_grad():
        out = diffusion_model.p_sample(
            unet,
            img,
            t,
            prot=protein_finger,
            w=5,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
        )
        img = out["sample"]

sample = img
sample = img.reshape(-1, 1, 32768)

mins = torch.tensor(np.load("data/smiles_mins_selfies.npy"), device=device).reshape(1, 1, -1)
maxes = torch.tensor(np.load("data/smiles_maxes_selfies.npy"), device=device).reshape(1, 1, -1)
sample_rescale = (((sample + 1) / 2) * (maxes - mins) + mins)
# print(sample_rescale.shape)

# Step 0.3.3: Convert from SELFIES back to SMILES
tokenizer = SelfiesTok.load("models/selfies_tok.json")
with torch.no_grad():
    decoded_smiles, _ = mol_model.decode_representation(sample_rescale.reshape(-1, max_seq_length, d_model), None, max_length=128, tokenizer=tokenizer)

def compute_qed_sas(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            qed = QED.qed(mol)
            sas = sascorer.calculateScore(mol)
            normalized_sas = 1 - sas / 10
            # print(smiles)
            # print(qed)
            # print(normalized_sas)
            return qed + normalized_sas
        else:
            return None
    except:
        return None

def select_best_smiles(smiles_list):
    # print(smiles_list)
    results = []
    for idx, smiles in enumerate(smiles_list):
        # print(f"SMILES: {smiles}")
        score = compute_qed_sas(smiles)
        # print(score)
        if score is not None:
            results.append((idx, score))

    # Sort the results by score in descending order and take the top 5
    results.sort(key=lambda x: x[1], reverse=True)
    top_5_indices = [a[0] for a in results[:5]]
    top_5_smiles = [a[1] for a in results[:5]]
    print(np.mean(top_5_smiles))
    return top_5_indices

smiles_list = []
for decode in decoded_smiles:
    predicted_selfie = tokenizer.decode(decode.detach().cpu().flatten().tolist(), skip_special_tokens=True)
    predicted_smile = sf.decoder(predicted_selfie)
    smiles_list.append(predicted_smile)

best5 = sample[select_best_smiles(smiles_list), :, :]
num_generations = 20
time = torch.tensor([num_advance], device=device).repeat(per_new, 1)
protein_finger = (protein_sequence,) * per_new
for generation in range(num_generations):
    all_samples = None
    all_smiles = []
    for idx, prev_sample in enumerate(best5):
        prev_sample = prev_sample.reshape(-1, 256, 128).repeat(per_new, 1, 1)
        img = diffusion_model.q_sample(prev_sample, time)
        shape = (per_new, 256, 128)
        indices = list(range(num_advance))[::-1]
        # Lazy import so that we don't depend on tqdm.
        from tqdm.auto import tqdm
        indices = tqdm(indices)
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = diffusion_model.p_sample(
                    unet,
                    img,
                    t,
                    prot=protein_finger,
                    w=5,
                    clip_denoised=True,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=None,
                )
                img = out["sample"]

        sample = img
        sample = img.reshape(-1, 1, 32768)

        mins = torch.tensor(np.load("data/smiles_mins_selfies.npy"), device=device).reshape(1, 1, -1)
        maxes = torch.tensor(np.load("data/smiles_maxes_selfies.npy"), device=device).reshape(1, 1, -1)
        sample_rescale = (((sample + 1) / 2) * (maxes - mins) + mins)
        # print(sample_rescale.shape)

        # Step 0.3.3: Convert from SELFIES back to SMILES
        tokenizer = SelfiesTok.load("models/selfies_tok.json")
        with torch.no_grad():
            decoded_smiles, _ = mol_model.decode_representation(sample_rescale.reshape(-1, max_seq_length, d_model), None, max_length=128, tokenizer=tokenizer)

        for decode in decoded_smiles:
            predicted_selfie = tokenizer.decode(decode.detach().cpu().flatten().tolist(), skip_special_tokens=True)
            predicted_smile = sf.decoder(predicted_selfie)
            all_smiles.append(predicted_smile)

        if idx == 0:
            all_samples = sample
        else:
            all_samples = torch.cat([all_samples, sample])

    best5 = all_samples[select_best_smiles(all_smiles), :, :]

mins = torch.tensor(np.load("data/smiles_mins_selfies.npy"), device=device).reshape(1, 1, -1)
maxes = torch.tensor(np.load("data/smiles_maxes_selfies.npy"), device=device).reshape(1, 1, -1)
best5_rescale = (((best5 + 1) / 2) * (maxes - mins) + mins)

evolved_smiles = []
tokenizer = SelfiesTok.load("models/selfies_tok.json")
with torch.no_grad():
    decoded_smiles, _ = mol_model.decode_representation(best5_rescale.reshape(-1, max_seq_length, d_model), None, max_length=128, tokenizer=tokenizer)

for decode in decoded_smiles:
    predicted_selfie = tokenizer.decode(decode.detach().cpu().flatten().tolist(), skip_special_tokens=True)
    predicted_smile = sf.decoder(predicted_selfie)
    evolved_smiles.append(predicted_smile)

print(evolved_smiles)