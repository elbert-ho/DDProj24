import random
from Bio import SeqIO

def extract_random_proteins(input_file, output_file, num_proteins):
    # Read all sequences from the input file
    sequences = list(SeqIO.parse(input_file, "fasta"))
    
    # Check if the input file has fewer sequences than requested
    if len(sequences) < num_proteins:
        raise ValueError(f"Input file only contains {len(sequences)} sequences, fewer than the requested {num_proteins}.")

    # Randomly sample the specified number of sequences
    random_sequences = random.sample(sequences, num_proteins)
    
    # Write the sampled sequences to the output file
    with open(output_file, "w") as output_handle:
        SeqIO.write(random_sequences, output_handle, "fasta")

# Example usage
input_file = "../data/uniprot.fasta"  # Replace with your input file path
output_file = "../data/uniprot1000.fasta"  # Replace with your desired output file path
num_proteins = 1000

extract_random_proteins(input_file, output_file, num_proteins)
print(f"Extracted {num_proteins} random protein sequences to {output_file}")
