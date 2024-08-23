import numpy as np

def calculate_bounding_box(pdb_file):
    x_coords = []
    y_coords = []
    z_coords = []

    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith(('ATOM', 'HETATM')):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(z)

    x_min, y_min, z_min = np.min(x_coords), np.min(y_coords), np.min(z_coords)
    x_max, y_max, z_max = np.max(x_coords), np.max(y_coords), np.max(z_coords)

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    center_z = (z_min + z_max) / 2.0

    size_x = x_max - x_min
    size_y = y_max - y_min
    size_z = z_max - z_min

    center = (center_x, center_y, center_z)
    sizes = (size_x, size_y, size_z)

    return center, sizes

# Example usage
pdb_file = '9ayg-edited.pdb'  # Replace with your file path
center, sizes = calculate_bounding_box(pdb_file)
print("Bounding Box Center:", center)
print("Bounding Box Sizes:", sizes)
