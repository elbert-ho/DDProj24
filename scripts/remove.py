import os

# Directory prefix
prefix = "../ligands_cl3_output"

# List of ligand files to delete (without the prefix)
ligand_files = [
    "ligand401.sdf",
    "ligand951.sdf",
    "ligand001.sdf",
    "ligand702.sdf",
    "ligand103.sdf",
    "ligand553.sdf",
    "ligand454.sdf",
    "ligand154.sdf",
    "ligand855.sdf",
    "ligand055.sdf",
    "ligand357.sdf",
    "ligand757.sdf",
    "ligand007.sdf",
    "ligand858.sdf",
    "ligand759.sdf",
    "ligand510.sdf",
    "ligand062.sdf",
    "ligand762.sdf",
    "ligand863.sdf",
    "ligand215.sdf",
    "ligand515.sdf",
    "ligand666.sdf",
    "ligand716.sdf",
    "ligand816.sdf",
    "ligand617.sdf",
    "ligand717.sdf",
    "ligand517.sdf",
    "ligand367.sdf",
    "ligand820.sdf",
    "ligand670.sdf",
    "ligand422.sdf",
    "ligand572.sdf",
    "ligand168.sdf",
    "ligand622.sdf",
    "ligand072.sdf",
    "ligand672.sdf",
    "ligand223.sdf",
    "ligand170.sdf",
    "ligand624.sdf",
    "ligand726.sdf",
    "ligand977.sdf",
    "ligand577.sdf",
    "ligand376.sdf",
    "ligand630.sdf",
    "ligand529.sdf",
    "ligand231.sdf",
    "ligand876.sdf",
    "ligand131.sdf",
    "ligand733.sdf",
    "ligand035.sdf",
    "ligand439.sdf",
    "ligand737.sdf",
    "ligand037.sdf",
    "ligand282.sdf",
    "ligand335.sdf",
    "ligand542.sdf",
    "ligand885.sdf",
    "ligand543.sdf",
    "ligand792.sdf",
    "ligand091.sdf",
    "ligand742.sdf",
    "ligand392.sdf",
    "ligand185.sdf",
    "ligand445.sdf",
    "ligand188.sdf",
    "ligand842.sdf",
    "ligand342.sdf"
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
