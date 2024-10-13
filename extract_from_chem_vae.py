from chemistry_vae2 import *
import json

content = open('logfile.dat', 'w')
content.close()
content = open('results.dat', 'w')
content.close()

if os.path.exists("settings2.yml"):
    settings = yaml.safe_load(open("settings2.yml", "r"))
else:
    print("Expected a file settings2.yml but didn't find it.")
    exit()

print('--> Acquiring data...')
type_of_encoding = settings['data']['type_of_encoding']
file_name_smiles = settings['data']['smiles_file']

print('Finished acquiring data.')

def get_selfie_and_smiles_encodings_for_dataset2(file_path):
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

    # Path to the saved SELFIES tokenizer file
    selfies_tokenizer_path = "data/selfies_tokenizer.json"

    # Load the SELFIES alphabet from the file
    with open(selfies_tokenizer_path, 'r') as f:
        selfies_alphabet = json.load(f)

    largest_smiles_len = len(max(smiles_list, key=len))

    print('--> Translating SMILES to SELFIES...')
    selfies_list = list(map(sf.encoder, smiles_list))
    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)

    print('Finished translating SMILES to SELFIES.')

    return selfies_list, selfies_alphabet, largest_selfies_len

print('Representation: SELFIES')
encoding_list, encoding_alphabet, largest_molecule_len = get_selfie_and_smiles_encodings_for_dataset2(file_name_smiles)
data = multiple_selfies_to_hot(encoding_list, largest_molecule_len,
                                    encoding_alphabet)

len_max_molec = data.shape[1]
len_alphabet = data.shape[2]
len_max_mol_one_hot = len_max_molec * len_alphabet

print(' ')
print(f"Alphabet has {len_alphabet} letters, "
        f"largest molecule is {len_max_molec} letters.")

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

vae_encoder.load_state_dict("saved_models//E.pt")
vae_decoder.load_state_dict("saved_models//D.pt")

vae_encoder.eval()
vae_decoder.eval()

# Function to save latent representations to a file
def save_latent_representations(latents, file_path):
    with open(file_path, 'wb') as f:
        np.save(f, latents)
    print(f"Latent representations saved to {file_path}")

# Store latent representations
latent_representations = []

# Process data in batches
print('Encoding molecules into latent space...')
num_batches = len(data) // batch_size

for i in range(num_batches + 1):
    batch_data = data[i * batch_size:(i + 1) * batch_size]
    
    if len(batch_data) == 0:
        continue

    batch_data_flattened = batch_data.reshape(len(batch_data), -1).astype(np.float32)
    batch_data_tensor = torch.tensor(batch_data_flattened).to(device)
    
    with torch.no_grad():
        z, _, _ = vae_encoder(batch_data_tensor)  # Only z is needed for later decoding
        latent_representations.append(z.cpu().numpy())

# Concatenate all latent representations into a single array
latent_representations = np.concatenate(latent_representations, axis=0)

# Save the latent representations for later use in diffusion model
save_latent_representations(latent_representations, "data/latent_representations.npy")

print("Latent encoding completed.")