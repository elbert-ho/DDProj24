def generate_ligand_list(file_name, n):
    with open(file_name, 'w') as file:
        for i in range(n):
            line = f"ligands_cl3_output/ligand{i:03}.sdf\n"
            file.write(line)

# Example usage
n = 1000  # Set n to the desired number of lines
generate_ligand_list("../ligands.txt", n)
