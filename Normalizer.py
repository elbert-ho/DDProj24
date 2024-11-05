import numpy as np
from scipy.stats import norm

class Normalizer:
    def __init__(self, mean_path="data/smiles_global_mean.npy", std_path="data/smiles_global_std.npy"):
        # Load the global mean and standard deviation from the specified files
        self.global_mean = np.load(mean_path).astype(np.float32)
        self.global_std = np.load(std_path).astype(np.float32)

    def normalize(self, sample):
        # Subtract the mean and divide by the standard deviation
        standardized = (sample - self.global_mean) / self.global_std
        # Apply the CDF function to get the value between 0 and 1
        cdf_value = norm.cdf(standardized).astype(np.float32)
        # Scale the value to be between -1 and 1
        scaled_value = 2 * cdf_value - 1
        return scaled_value

    def denormalize(self, scaled_value):
        scaled_value = scaled_value.astype(np.float64)
        # Reverse the scaling to get back to the CDF value (0 to 1)
        cdf_value = (scaled_value + 1) / 2
        clipping = 9.76e-10
        # clipping = 1e-9
        cdf_value = np.clip(cdf_value, clipping, 1 - clipping)
        # print(cdf_value.max())
        # print(cdf_value.min())

        # Apply the inverse CDF (PPF) to get the standardized value
        standardized = norm.ppf(cdf_value).astype(np.float32)
        # Multiply by the standard deviation and add the mean to return to the original scale
        sample = standardized * self.global_std + self.global_mean
        # exit()
        return sample
