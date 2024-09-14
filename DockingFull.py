import os
import subprocess
from ClearFolders import *

conda_path = "/home/student/miniconda3/condabin/conda"
python_path = "/home/student/miniconda3/envs/unidock_env/bin/python"
env_name = "unidock_env"

# Parameters
# ws = [0, 0.5, 5, 10, 50, 100, 200]
ws = [5]
for w in ws:
# w = 0
    num_proteins = 100
    # protein_file = "5f1a.pdbqt"
    # cx = 40.544
    # cy = 24.996
    # cz = 241.266

    protein_file = "9ayh.pdbqt"
    cx = 139.842
    cy = 124.090
    cz = 148.222

    # Assuming output is 21 for testing purposes
    # output = 20

    # Run DockingTest.py

    delete_sdf_files("ligands_cl3")
    delete_sdf_files("ligands_cl3_filtered")

    result = subprocess.run(
        ["python", "DockingTest.py", "--w", str(w), "--num_proteins", str(num_proteins)], 
        text=True
    )

    folder_path = "ligands_cl3_filtered"
    output = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])

    # Clear folders
    subprocess.run(["python", "ClearFolders.py"], stdout=subprocess.DEVNULL)

    # Prepare ligands
    subprocess.run(["python", "scripts/LigandTextFile.py", "--n", str(num_proteins), "--output_dir", "ligands_cl3"], stdout=subprocess.DEVNULL)
    # exit()
    # Activate environment and prepare ligands using UniDockTools
    subprocess.run([
        "wsl", conda_path, "run", "-n", env_name, "unidocktools", "ligandprep", "-i", "ligands.txt", "-sd", "ligands_cl3_output"
    ], stdout=subprocess.DEVNULL)

    # Prepare ligands for docking
    subprocess.run(["python", "scripts/LigandTextFile.py", "--n", str(num_proteins), "--output_dir", "ligands_cl3_output"], stdout=subprocess.DEVNULL)

    # Run UniDockTools docking pipeline with piping
    # Define the pipeline command to run within WSL
    pipeline_command_str = (
        f"{conda_path} run -n {env_name} unidocktools unidock_pipeline "
        f"-r {protein_file} -i ligands.txt -sd docking_output_cl3 "
        f"-cx {cx} -cy {cy} -cz {cz}"
    )

    # # Run the command with piping to remove_torsion.py within WSL
    subprocess.run(
        f"wsl bash -c \"{pipeline_command_str} | {python_path} remove_torsion.py > /dev/null 2>&1\"", 
        shell=True, check=True
    )

    # Run the command again without piping within WSL
    subprocess.run(f"wsl bash -c \"{pipeline_command_str} > /dev/null 2>&1\"", shell=True, check=True)

    # Print benchmark
    subprocess.run(["python", "DockingPrintBench.py"])

    print("-----------------------------------")

    # Re-run the same steps with the updated output
    num_proteins = output

    subprocess.run(["python", "ClearFolders.py"], stdout=subprocess.DEVNULL)
    subprocess.run(["python", "scripts/LigandTextFile.py", "--n", str(num_proteins), "--output_dir", "ligands_cl3_filtered"], stdout=subprocess.DEVNULL)
    subprocess.run([
        "wsl", conda_path, "run", "-n", env_name, "unidocktools", "ligandprep", "-i", "ligands.txt", "-sd", "ligands_cl3_output"
    ], stdout=subprocess.DEVNULL)
    subprocess.run(["python", "scripts/LigandTextFile.py", "--n", str(num_proteins), "--output_dir", "ligands_cl3_output"], stdout=subprocess.DEVNULL)

    # Run the command with piping to remove_torsion.py within WSL
    subprocess.run(
        f"wsl bash -c \"{pipeline_command_str} | {python_path} remove_torsion.py > /dev/null 2>&1\"", 
        shell=True, check=True
    )

    # Run the command again without piping within WSL
    subprocess.run(f"wsl bash -c \"{pipeline_command_str} > /dev/null 2>&1\"", shell=True, check=True)

    # Print benchmark again
    subprocess.run(["python", "DockingPrintBench.py"])
