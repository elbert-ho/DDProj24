import csv
import re

def clean_header(header):
    # Use regular expression to extract the actual column names
    clean_header = []
    for col in header:
        match = re.search(r"'(\w+)'", col)
        if match:
            clean_header.append(match.group(1))
        else:
            clean_header.append(col.strip())
    return clean_header

def clean_csv(file_path, output_file_path):
    with open(file_path, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        
        # Clean the header
        cleaned_header = clean_header(header)
        
        with open(output_file_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(cleaned_header)  # Write the cleaned header
            for row in reader:
                writer.writerow(row)  # Write the remaining rows

# Example usage
input_file = '../data/smiles_10000_with_props_full.csv'
output_file = '../data/smiles_10000_with_props_full_2.csv'
clean_csv(input_file, output_file)

print(f"Cleaned CSV saved to {output_file}")