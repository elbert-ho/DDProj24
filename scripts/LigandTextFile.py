import argparse

def generate_ligand_list(output_dir, file_name, n):
    with open(file_name, 'w') as file:
        for i in range(n):
            line = f"{output_dir}/ligand{i:03}.sdf\n"
            file.write(line)

# Argument parser to allow command line options
parser = argparse.ArgumentParser(description="Generate a list of ligand files.")
parser.add_argument("--output_dir", type=str, default="ligands_cl3_output", help="Directory where ligand SDF files are stored")
parser.add_argument("--n", type=int, default=100, help="Number of ligands to list")

args = parser.parse_args()

# Example usage
generate_ligand_list(args.output_dir, "ligands.txt", args.n)
