import pandas as pd

# Define the path to your TSV file
file_path = "/Users/Elbert/Downloads/BindingDB_All.tsv"

# Read the first 1 million rows of the TSV file
df = pd.read_csv(file_path, sep='\t', nrows=500000)

# Save the extracted rows to a new TSV file
output_path = "/Users/Elbert/Downloads/BindingDB_All.tsv"
df.to_csv(output_path, sep='\t', index=False)

print("First 1 million rows have been extracted and saved.")
