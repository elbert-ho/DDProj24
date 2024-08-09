import pandas as pd
import selfies as sf
from SelfiesTok import SelfiesTok

smiles = pd.read_csv("protein_drug_pairs_with_sequences_cleaned")["SMILES String"].to_list()
