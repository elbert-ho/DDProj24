import os
import re
import sys

def parse_ligand_files(log_content):
    # Regular expression to find ligands with "exceed max torsion counts"
    pattern = r'ligand\d+\.sdf'
    
    # Find all matches in the log content
    return re.findall(pattern, log_content)

def delete_ligand_files(ligand_files, directory_prefix):
    for ligand in ligand_files:
        file_path = os.path.join(directory_prefix, ligand)
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    # Read log content from stdin (assuming the log is piped into this script)
    log_content = sys.stdin.read()
    
    # Extract ligand files that exceed max torsion counts
    ligand_files = parse_ligand_files(log_content)
    
    # print(ligand_files[0])
    # exit()

    # Directory prefix (you can adjust this as needed)
    prefix = "ligands_cl3_output"
    
    # Delete the files
    delete_ligand_files(ligand_files, prefix)
