import os

# Directory prefix
prefix = "../ligands_cl3_output"

# List of ligand files to delete (without the prefix)
ligand_files = [
    "ligand454.sdf",
    "ligand304.sdf",
    "ligand904.sdf",
    "ligand906.sdf",
    "ligand557.sdf",
    "ligand907.sdf",
    "ligand359.sdf",
    "ligand009.sdf",
    "ligand560.sdf",
    "ligand110.sdf",
    "ligand963.sdf",
    "ligand363.sdf",
    "ligand913.sdf",
    "ligand615.sdf",
    "ligand365.sdf",
    "ligand868.sdf",
    "ligand219.sdf",
    "ligand370.sdf",
    "ligand023.sdf",
    "ligand724.sdf",
    "ligand226.sdf",
    "ligand027.sdf",
    "ligand028.sdf",
    "ligand280.sdf",
    "ligand130.sdf",
    "ligand281.sdf",
    "ligand633.sdf",
    "ligand385.sdf",
    "ligand531.sdf",
    "ligand533.sdf",
    "ligand684.sdf",
    "ligand638.sdf",
    "ligand140.sdf",
    "ligand893.sdf",
    "ligand238.sdf",
    "ligand292.sdf",
    "ligand296.sdf"
]

# Delete the files
for ligand in ligand_files:
    file_path = os.path.join(prefix, ligand)
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")
