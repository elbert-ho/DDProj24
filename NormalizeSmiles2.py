import numpy as np

# Load the SMILES data
smiles = np.load("data/smiles_output_selfies2.npy")

# Step 1: Compute the global mean and standard deviation
global_mean = np.load("data/smiles_global_mean.npy")
global_std = np.load("data/smiles_global_std.npy")
global_min = np.load("data/smiles_global_min.npy")
global_max = np.load("data/smiles_global_max.npy")

# global_mean = np.mean(smiles)
# global_std = np.std(smiles)

# Normalize the data to have mean 0 and variance 1
smiles_normalized = (smiles - global_mean) / global_std

# Step 2: Compute the global min and max from the normalized data
# global_min = np.min(smiles_normalized)
# global_max = np.max(smiles_normalized)

# Linearly rescale the data between -1 and 1
smiles_rescaled = 2 * (smiles_normalized - global_min) / (global_max - global_min) - 1

# Save the 4 values for reversibility
# np.save("data/smiles_global_mean.npy", global_mean)
# np.save("data/smiles_global_std.npy", global_std)
# np.save("data/smiles_global_min.npy", global_min)
# np.save("data/smiles_global_max.npy", global_max)

# Save the rescaled data
np.save("data/smiles_output_selfies_normal2.npy", smiles_rescaled)

print("Normalization and rescaling complete. Data and parameters saved.")
