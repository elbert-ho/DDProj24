import pandas as pd
from Bio import SeqIO

# Load the CSV file
csv_file = '../data/protein_drug_pairs_with_sequences_and_smiles.csv'
csv_data = pd.read_csv(csv_file)

# Load the FASTA file
fasta_file = '../data/uniprot1000.fasta'
fasta_sequences = SeqIO.parse(open(fasta_file), 'fasta')

# Extract protein sequences from the CSV file
csv_proteins = set(csv_data['Protein Sequence'].tolist())

# Function to check if a protein sequence is in the CSV
def is_in_csv(sequence, csv_proteins):
    return sequence in csv_proteins

# Create a list to hold the sequences not in the CSV
sequences_not_in_csv = []

# Iterate through the FASTA file and check if each sequence is in the CSV
for record in fasta_sequences:
    if not is_in_csv(str(record.seq), csv_proteins):
        sequences_not_in_csv.append(record)

# Write the sequences not in the CSV to a new FASTA file
output_fasta = '../data/test_prots1K.fasta'
with open(output_fasta, 'w') as output_handle:
    SeqIO.write(sequences_not_in_csv, output_handle, 'fasta')

print(f"Number of proteins written to the new FASTA file: {len(sequences_not_in_csv)}")