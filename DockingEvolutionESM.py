# Generate 1000 molecules

# maybe check w but doubtful it will do anything at all

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
from MolLoaderSelfiesFinal import SMILESDataset
from Normalizer import  Normalizer

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
mol_model.load_state_dict(torch.load('models/selfies_transformer_final_bpe.pt', map_location=device))

checkpoint = torch.load('checkpoint.pt', map_location=device)
unet.load_state_dict(checkpoint["state_dict"])

unet, mol_model = unet.to(device), mol_model.to(device)

# REFERENCE SAMPLE SHOULD BE UNNORMALIZED
dataset = SMILESDataset("data/pd_truncated_final_1.csv", tokenizer_path="models/selfies_tokenizer_final.json", unicode_path="models/unicode_mapping.json", props=False)

# reference_sample = torch.tensor(np.load("data/ref_sample.npy"), device=device).reshape(1, 1, 32768)
# reference_sample = np.load("data/smiles_output_selfies_normal2.npy")[0].reshape(1, 1, 32768)
# protein_ids = torch.tensor(np.load("data/input_ids2.npy")[0], dtype=torch.int32).reshape(1, -1).to(device)
# protein_atts = torch.tensor(np.load("data/attention_masks2.npy")[0], dtype=torch.int32).reshape(1, -1).to(device)

reference_sample = np.load("data/mol_19.npy")[0].reshape(1, 1, 32768)
protein_ids = torch.tensor(np.load("data/mol_19_ids.npy")[0], dtype=torch.int32).reshape(1, -1).to(device)
protein_atts = torch.tensor(np.load("data/mol_19_masks.npy")[0], dtype=torch.int32).reshape(1, -1).to(device)

# reference_sample = torch.tensor(np.load("data/smiles_output_selfies_normal.npy")[9653], device=device).reshape(1, 1, 32768)

# Get Reference SMILES
# global_mean = np.load("data/smiles_global_mean.npy")
# global_std = np.load("data/smiles_global_std.npy")
# global_min = np.load("data/smiles_global_min.npy")
# global_max = np.load("data/smiles_global_max.npy")

# print(f"dist {global_max - global_min}")

# print(reference_sample[0])

normalizer = Normalizer()

# reference_sample_rescale = ((reference_sample + 1) / 2) 
# print(reference_sample_rescale[0])
# reference_sample_rescale = reference_sample_rescale * (global_max - global_min) + global_min
# print(reference_sample_rescale[0])
# Step 2: Undo the Z-score normalization to restore the original data
# reference_sample_rescale = torch.tensor(reference_sample_rescale * global_std + global_mean, device=device)
# print(reference_sample_rescale[0])
# exit()

reference_sample_rescale = torch.tensor(normalizer.denormalize(reference_sample), device=device)

# print(reference_sample_rescale.dtype)

# reference_sample_rescale = torch.tensor(np.load("data/smiles_output_selfies2.npy"), device=device)[0].reshape(1, 1, 32768)

# print(reference_sample_rescale2.dtype)

# def find_distant_elements(tensor1, tensor2, threshold):
#     # Calculate the absolute difference between the tensors
#     difference = torch.abs(tensor1 - tensor2)
    
#     # Find the indices where the difference exceeds the threshold
#     distant_elements_indices = torch.nonzero(difference > threshold, as_tuple=True)
    
#     # Extract the values from tensor1 and tensor2 using the found indices
#     tensor1_distant_values = tensor1[distant_elements_indices]
#     tensor2_distant_values = tensor2[distant_elements_indices]
    
#     # Return the indices, and the values from both tensors at those indices
#     return distant_elements_indices, tensor1_distant_values, tensor2_distant_values

# indices, values1, values2 = find_distant_elements(reference_sample_rescale, reference_sample_rescale2, .1)
# print("Indices of distant elements:", indices)
# print("Values from tensor1 at those indices:", values1)
# print("Values from tensor2 at those indices:", values2)



# print(torch.allclose(reference_sample_rescale, reference_sample_rescale2))

# Step 0.3.3: Convert from SELFIES back to SMILES
# tokenizer = SelfiesTok.load("models/selfies_tok.json")
with torch.no_grad():
    decoded_smiles, _ = mol_model.decode_representation(reference_sample_rescale.reshape(-1, max_seq_length, d_model), None, max_length=128, tokenizer=dataset.tokenizer)

predicted_selfie = dataset.decode(decoded_smiles[0].detach().cpu().flatten().tolist())
reference_smile = sf.decoder(predicted_selfie)

reference_mol = Chem.MolFromSmiles(reference_smile)
reference_fp = AllChem.GetMorganFingerprintAsBitVect(reference_mol, radius=2, nBits=2048)

# print(predicted_selfie)
# print(reference_smile)

# exit()

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

num_gen = 25
reference_sample_flattened = torch.tensor(reference_sample.reshape(1, -1), device=device)
reference_sample = torch.tensor(reference_sample, device=device)
reference_sample = reference_sample.reshape(-1, 256, 128).repeat(num_gen, 1, 1)
# print(reference_sample[0])

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

# tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

# df = pd.read_csv("data/protein_drug_pairs_with_sequences_and_smiles_cleaned2.csv")
# protein_strings = df["Protein Sequence"].to_list()

# protein_strings = "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQCSGVTFQ"
# xf_in = tokenizer(protein_strings, return_tensors="pt", padding=True, truncation=True, max_length=1026)
# protein_ids = xf_in["input_ids"]
# protein_atts = xf_in["attention_mask"]

protein_ids_repeat = protein_ids.repeat(num_gen, 1).to("cuda")
protein_atts_repeat = protein_atts.repeat(num_gen, 1).to("cuda")

if False:
    # ws = [10, 20, 50]
    # for w_test in ws:
        # print(w_test)
        # step = 999
    for step in range(999, 1000, 100):
    # step = 999
    # for step in range(10, 101, 10):
    # for step in range(1, 11, 1):
        time = torch.tensor([step], device=device).repeat(num_gen, 1)

        img = diffusion_model.q_sample(reference_sample, time)

        # print(reference_sample[0][0])
        # print(img[0][0])

        shape = (num_gen, 256, 128)
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
                    w=3,
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
        sample = sample.cpu().detach().numpy()
        # reference_sample_np = reference_sample.reshape(1,-1).cpu().detach().numpy()
        # sample_rescale = ((sample + 1) / 2) * (global_max - global_min) + global_min
        # Step 2: Undo the Z-score normalization to restore the original data
        # sample_rescale = torch.tensor(sample_rescale * global_std + global_mean, device=device)
        
        sample_rescale = torch.tensor(normalizer.denormalize(sample), device=device)

        # print(global_min)
        # print(global_max)
        # print(global_std)
        # print(global_mean)
        # print(sample[0])
        # print(sample_rescale[0])
        test_sample_flatten = sample[0].reshape(1, -1)

        # print(test_sample_flatten)

        # HERE
        # print(reference_sample_flattened)

        distance = torch.sqrt(torch.sum(torch.pow(torch.subtract(torch.tensor(test_sample_flatten, device=device), reference_sample_flattened), 2)) / 32768)  
        print(distance)

        # print(reference_sample_flattened)
        # print(test_sample_flatten)

        # print(sample_rescale.shape)
        # sample = torch.tensor(sample, device=device)
        # infinity_mask = torch.isinf(sample_rescale)
        # values_at_infinity = sample[infinity_mask]
        # print(values_at_infinity) 

        # print(torch.isinf(sample_rescale).any())
        # exit()

        # Step 0.3.3: Convert from SELFIES back to SMILES
        with torch.no_grad():
            decoded_smiles, _ = mol_model.decode_representation(sample_rescale.reshape(-1, max_seq_length, d_model), None, max_length=128, tokenizer=dataset.tokenizer)

        qeds = []
        sass = []
        sims = []
        final_smiles = []
        for decode in decoded_smiles:
            predicted_selfie = dataset.decode(decode.detach().cpu().flatten().tolist())
            predicted_smile = sf.decoder(predicted_selfie)
            
            # print(predicted_selfie)

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

# FIX THIS CODE LATER

num_advance = 75

per_new = 50
num_gen = 50
reference_sample = reference_sample[0].reshape(1, 256, 128).repeat(num_gen, 1, 1)
protein_ids_repeat = protein_ids[0].repeat(num_gen, 1).to("cuda")
protein_atts_repeat = protein_atts[0].repeat(num_gen, 1).to("cuda")

time = torch.tensor([num_advance], device=device).repeat(num_gen, 1)
# Generation 0
img = diffusion_model.q_sample(reference_sample, time)
shape = (num_gen, 256, 128)
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
            prot=protein_ids_repeat,
            attn=protein_atts_repeat,
            w=0,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
        )
        img = out["sample"]

sample = img
sample = img.reshape(-1, 1, 32768)

sample = sample.cpu().detach().numpy()
# reference_sample_np = reference_sample.reshape(1,-1).cpu().detach().numpy()
# sample_rescale = ((sample + 1) / 2) * (global_max - global_min) + global_min
# Step 2: Undo the Z-score normalization to restore the original data
# sample_rescale = torch.tensor(sample_rescale * global_std + global_mean, device=device)

sample_rescale = torch.tensor(normalizer.denormalize(sample), device=device)

# print(sample_rescale.shape)

# Step 0.3.3: Convert from SELFIES back to SMILES
with torch.no_grad():
    decoded_smiles, _ = mol_model.decode_representation(sample_rescale.reshape(-1, max_seq_length, d_model), None, max_length=128, tokenizer=dataset.tokenizer)

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
    predicted_selfie = dataset.decode(decode.detach().cpu().flatten().tolist())
    predicted_smile = sf.decoder(predicted_selfie)
    smiles_list.append(predicted_smile)

best5 = sample[select_best_smiles(smiles_list), :, :]
num_generations = 1
time = torch.tensor([num_advance], device=device).repeat(per_new, 1)
for generation in range(num_generations):
    flag = False
    while not flag:
        all_samples = None
        all_smiles = []
        for idx, prev_sample in enumerate(best5):
            prev_sample = torch.tensor(prev_sample, device=device)
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
                        prot=protein_ids_repeat,
                        attn=protein_atts_repeat,
                        w=0,
                        clip_denoised=True,
                        denoised_fn=None,
                        cond_fn=None,
                        model_kwargs=None,
                    )
                    img = out["sample"]

            sample = img
            sample = img.reshape(-1, 1, 32768)
            sample = sample.cpu().detach().numpy()
            # reference_sample_np = reference_sample.reshape(1,-1).cpu().detach().numpy()
            # sample_rescale = ((sample + 1) / 2) * (global_max - global_min) + global_min
            # Step 2: Undo the Z-score normalization to restore the original data
            # sample_rescale = torch.tensor(sample_rescale * global_std + global_mean, device=device)

            sample_rescale = torch.tensor(normalizer.denormalize(sample), device=device)

            # print(sample_rescale.shape)

            # Step 0.3.3: Convert from SELFIES back to SMILES
            with torch.no_grad():
                decoded_smiles, _ = mol_model.decode_representation(sample_rescale.reshape(-1, max_seq_length, d_model), None, max_length=128, tokenizer=dataset.tokenizer)
            indices = []
            idx0 = 0
            for decode in decoded_smiles:
                predicted_selfie = dataset.decode(decode.detach().cpu().flatten().tolist())
                predicted_smile = sf.decoder(predicted_selfie)
                mol = Chem.MolFromSmiles(predicted_smile)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                sim = FingerprintSimilarity(reference_fp, fp)
                if(sim >= 0.2):
                    all_smiles.append(predicted_smile)
                    indices.append(idx0)
                
                idx0 += 1

            sample = sample[indices]

            if len(sample) == 0:
                continue

            if all_samples is None:
                all_samples = sample
            else:
                all_samples = np.concatenate([all_samples, sample])

        if all_samples is not None:
            flag = True
            best5 = all_samples[select_best_smiles(all_smiles), :, :]

best5_rescale = torch.tensor(normalizer.denormalize(best5), device=device)

evolved_smiles = []
with torch.no_grad():
    decoded_smiles, _ = mol_model.decode_representation(sample_rescale.reshape(-1, max_seq_length, d_model), None, max_length=128, tokenizer=dataset.tokenizer)

for decode in decoded_smiles:
    predicted_selfie = dataset.decode(decode.detach().cpu().flatten().tolist())
    predicted_smile = sf.decoder(predicted_selfie)
    evolved_smiles.append(predicted_smile)

for evolved_smile in evolved_smiles:
    mol = Chem.MolFromSmiles(evolved_smile)
    # mol = Chem.AddHs(mol)
    # Filter for QED at least 0.5, SAS below 5
    qed = QED.qed(mol)
    sas = sascorer.calculateScore(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    sim = FingerprintSimilarity(reference_fp, fp)
    print(f"SMILES {evolved_smile} QED: {qed} SAS: {sas} SIM: {sim}")


# print(evolved_smiles)