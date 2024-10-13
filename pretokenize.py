from transformers import EsmTokenizer, EsmModel, EsmConfig
import pandas as pd
import numpy as np

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

# df = pd.read_csv("data/protein_drug_pairs_with_sequences_and_smiles_cleaned2.csv")
# protein_strings = df["Protein Sequence"].to_list()

protein_strings = ["", "MASWSHPQFEKGGGARGGSGGGSWSHPQFEKGFDYKDDDDKGTMTEGARAADEVRVPLGAPPPGPAALVGASPESPGAPGREAERGSELGVSPSESPAAERGAELGADEEQRVPYPALAATVFFCLGQTTRPRSWCLRLVCNPWFEHVSMLVIMLNCVTLGMFRPCEDVECGSERCNILEAFDAFIFAFFAVEMVIKMVALGLFGQKCYLGDTWNRLDFFIVVAGMMEYSLDGHNVSLSAIRTVRVLRPLRAINRVPSMRILVTLLLDTLPMLGNVLLLCFFVFFIFGIVGVQLWAGLLRNRCFLDSAFVRNNNLTFLRPYYQTEEGEENPFICSSRRDNGMQKCSHIPGRRELRMPCTLGWEAYTQPQAEGVGAARNACINWNQYYNVCRSGDSNPHNGAINFDNIGYAWIAIFQVITLEGWVDIMYYVMDAHSFYNFIYFILLIIVGSFFMINLCLVVIATQFSETKQRESQLMREQRARHLSNDSTLASFSEPGSCYEELLKYVGHIFRKVKRRSLRLYARWQSRWRKKVDPGWMGRLWVTFSGKLRRIVDSKYFSRGIMMAILVNTLSMGVEYHEQPEELTNALEISNIVFTSMFALEMLLKLLACGPLGYIRNPYNIFDGIIVVISVWEIVGQADGGLSVLRTFRLLRVLKLVRFLPALRRQLVVLVKTMDNVATFCTLLMLFIFIFSILGMHLFGCKFSLKTDTGDTVPDRKNFDSLLWAIVTVFQILTQEDWNVVLYNGMASTSSWAALYFVALMTFGNYVLFNLLVAILVEGFQAEGDANRSDTDEDKTSVHFEEDFHKLRELQTTELKMCSLAVTPNGHLEGRGSLSPPLIMCTAATPMPTPKSSPFLDAAPSLPDSRRGSSSSGDPPLGDQKPPASLRSSPCAPWGPSGAWSSRRSSWSSLGRAPSLKRRGQCGERESLLSGEGKGSTDDEAEDGRAAPGPRATPLRRAESLDPRPLRPAALPPTKCRDRDGQVVALPSDFFLRIDSHREDAAELDDDSEDSCCLRLHKVLEPYKPQWCRSREAWALYLFSPQNRFRVSCQKVITHKMFDHVVLVFIFLNCVTIALERPDIDPGSTERVFLSVSNYIFTAIFVAEMMVKVVALGLLSGEHAYLQSSWNLLDGLLVLVSLVDIVVAMASAGGAKILGVLRVLRLLRTLRPLRVISRAPGLKLVVETLISSLRPIGNIVLICCAFFIIFGILGVQLFKGKFYYCEGPDTRNISTKAQCRAAHYRWVRRKYNFDNLGQALMSLFVLSSKDGWVNIMYDGLDAVGVDQQPVQNHNPWMLLYFISFLLIVSFFVLNMFVGVVVENFHKCRQHQEAEEARRREEKRLRRLERRRRSTFPSPEAQRRPYYADYSPTRRSIHSLCTSHYLDLFITFIICVNVITMSMEHYNQPKSLDEALKYCNYVFTIVFVFEAALKLVAFGFRRFFKDRWNQLDLAIVLLSLMGITLEEIEMSAALPINPTIIRIMRVLRIARVLKLLKMATGMRALLDTVVQALPQVGNLGLLFMLLFFIYAALGVELFGRLECSEDNPCEGLSRHATFSNFGMAFLTLFRVSTGDNWNGIMKDTLRECSREDKHCLSYLPALSPVYFVTFVLVAQFVLVNVVVAVLMKHLEESNKEAREDAELDAEIELEMAQGPGSARRVDADRPPLPQESPGARDAPNLVARKVSVSRMLSLPNDSYMFRPVVPASAPHPRPLQEVEMETYGAGTPLGSVASVHSPPAESCASLQIPLAVSSPARSGEPLHALSPRGTARSPSLSRLLCRQEAVHTDSLEGKIDSPRDTLDPAEPGEKTPVRPVTQGGSLQSPPRSPRPASVRTRKHTFGQRCVSSRPAAPGGEEAEASDPADEEVSHITSSACPWQPTAEPHGPEASPVAGGERDLRRLYSVDAQGFLDKPGRADEQWRPSAELGSGEPGEAKAWGPEAEPALGARRKKKMSPPCISVEPPAEDEGSARPSAAEGGSTTLRRRTPSCEATPHRDSLEPTEGSGAGGDPAAKGERWGQASCRAEHLTVPSFAFEPLDLGVPSGDPFLDGSHSVTPESRASSSGAIVPLEPPESEPPMPVGDPPEKRRGLYLTVPQCPLEKPGSPSATPAPGGGADDPV"]
xf_in = tokenizer(protein_strings, return_tensors="pt", padding=True, truncation=True, max_length=1026)
input_ids = xf_in["input_ids"].numpy()
attention_mask = xf_in["attention_mask"].numpy()

print(input_ids)
print(attention_mask)

# np.save("data/input_ids2.npy", input_ids)
# np.save("data/attention_masks2.npy", attention_mask)
