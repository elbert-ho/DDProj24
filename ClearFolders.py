import os
import glob

# Define the directories
dir1 = "ligands_cl3_output"
dir2 = "docking_output_cl3"
# List of directories
directories = [dir1, dir2]
# directories = [dir2]

# Function to delete .sdf files
def delete_sdf_files(directory):
    if os.path.exists(directory):
        files = glob.glob(os.path.join(directory, "*.sdf"))
        for file in files:
            try:
                os.remove(file)
                # print(f"Deleted: {file}")
            except Exception as e:
                print(f"Failed to delete {file}: {e}")
    else:
        print(f"Directory does not exist: {directory}")

# Run the deletion process for each directory
for directory in directories:
    delete_sdf_files(directory)

print("Deletion process completed.")
