from Bio import SeqIO

def count_proteins_in_fasta(file_path):
    count = 0
    for record in SeqIO.parse(file_path, "fasta"):
        count += 1
    return count

# Example usage
file_path = "../data/uniprot.fasta"  # Replace with your file path
num_proteins = count_proteins_in_fasta(file_path)
print(f"Number of proteins in the FASTA file: {num_proteins}")
