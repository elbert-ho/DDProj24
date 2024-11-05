import numpy as np
from Normalizer import Normalizer

# Load the SMILES data
smiles = np.load("data/smiles_output_selfies.npy")
# print(smiles)

# print(smiles.shape)

normalizer = Normalizer()
smiles_rescaled = normalizer.normalize(smiles)

# print(smiles_rescaled.shape)

# print(smiles_rescaled)
# print(normalizer.denormalize(smiles_rescaled))
# Save the rescaled data
# print(smiles.dtype)
# print(smiles_rescaled.dtype)

np.save("data/smiles_output_selfies_normal.npy", smiles_rescaled.astype(np.float32))

print("Normalization and rescaling complete. Data and parameters saved.")
