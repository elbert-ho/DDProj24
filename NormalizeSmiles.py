import numpy as np

smiles = np.load("data/smiles_output_selfies2.npy")
smiles_maxes = np.amax(smiles, axis=0).reshape(1, -1)
smiles_mins = np.amin(smiles, axis=0).reshape(1, -1)
smiles_normalized = 2 * (smiles - smiles_mins) / (smiles_maxes - smiles_mins) - 1

np.save("data/smiles_output_selfies_normal2.npy", smiles_normalized)
np.save("data/smiles_maxes_selfies.npy", smiles_maxes)
np.save("data/smiles_mins_selfies.npy", smiles_mins)