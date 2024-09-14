import os
from rdkit import Chem
import numpy as np

def extract_top_docking_scores(sdf_folder):
    top_scores = []

    for file_name in os.listdir(sdf_folder):
        if file_name.endswith('.sdf'):
            file_path = os.path.join(sdf_folder, file_name)
            suppl = Chem.SDMolSupplier(file_path)
            file_scores = []
            for mol in suppl:
                if mol is not None:
                    # Extract the docking score
                    score = float(mol.GetProp('docking_score'))
                    file_scores.append(score)
            if file_scores:
                top_scores.append(min(file_scores))  # Get the best (lowest) score in the file

            if min(file_scores) < -7:
                print(file_name)

    return top_scores

def calculate_statistics(scores):
    scores.sort()
    average_score = sum(scores) / len(scores)
    top_10_percent = scores[:max(1, int(0.1 * len(scores)))]  # Ensure at least one score is returned

    return average_score, top_10_percent

def main():
    sdf_folder = "docking_output_cl3"  # Replace with the path to your folder
    top_scores = extract_top_docking_scores(sdf_folder)

    if not top_scores:
        print("No docking scores found.")
        return
    index_min = np.argmin(top_scores)
    average_score, top_10_percent = calculate_statistics(top_scores)

    print(f"Average Docking Score: {average_score}")
    print(f"Average Top 10% {sum(top_10_percent) / len(top_10_percent)}")
    print(f"Smallest: {index_min}")
    print(f"Smallest 10% {top_10_percent}")
if __name__ == "__main__":
    main()
