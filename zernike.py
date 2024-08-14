import numpy as np
import matplotlib.pyplot as plt
import json
from aotools.functions.zernike import zernikeArray, phaseFromZernikes


def create_mask(img_w, img_h, radius, center_x, center_y):
    mask = np.zeros((img_h, img_w), dtype=np.bool)
    for i in range(img_h):
        for j in range(img_w):
            if (i - center_y)**2 + (j - center_x)**2 <= radius**2:
                mask[i, j] = True
    return mask

# Function to crop the masked array to its unmasked region
def crop_to_unmasked(masked_array):
    mask = masked_array.mask
    if not np.any(~mask):
        raise ValueError("No unmasked regions in the array.")
    
    # Find the bounding box of the unmasked region
    unmasked_indices = np.where(~mask)
    min_row, max_row = np.min(unmasked_indices[0]), np.max(unmasked_indices[0])
    min_col, max_col = np.min(unmasked_indices[1]), np.max(unmasked_indices[1])
    
    # Crop the array to the bounding box
    cropped_array = masked_array[min_row:max_row + 1, min_col:max_col + 1]
    
    return cropped_array

test_id = input("Enter test ID: ")
TEST = int(test_id)
print(f"Test {TEST} selected")
# Define the paths
measurement_data_path = f'Measurement data/Test {TEST}/'
calibration_data_path = f'Calibration data/Test {TEST}/'
params_path = f'Results/Test {TEST}/params.json'
results_path = f'Results/Test {TEST}/'

# Load parameters from JSON file
with open(params_path, "r") as file:
    json_data = json.load(file)


mask_radius = json_data['mask_radius']
mask_center_x = json_data['mask_center_x']
mask_center_y = json_data['mask_center_y']


surface = np.load(results_path+"surface.npy")


# Create the mask
h, w = surface.shape
mask = create_mask(w, h, mask_radius, mask_center_x, mask_center_y)

# Apply the mask to the surface
surface = np.ma.array(surface, mask=~mask)

surface = crop_to_unmasked(surface)

# Define the parameters
N = surface.shape[0]  # Size of the array
noll_index = 10  # Number of Zernike modes to use

# Create Zernike polynomials
zernike_polynomials = zernikeArray(noll_index, N)
print(zernike_polynomials.shape)

# Flatten the height map and Zernike polynomials
surface_flat = surface.flatten()
zernike_flat = zernike_polynomials.reshape((noll_index, -1))
print(zernike_flat.shape)

# Fit the Zernike coefficients
A = np.vstack(zernike_flat).T
coefficients, _, _, _ = np.linalg.lstsq(A, surface_flat, rcond=None)
print(coefficients)

# Plot the histogram of the value associated with each Zernike mode
modes = np.arange(1, noll_index + 1)
plt.bar(modes, coefficients)
plt.xlabel('Zernike Mode')
plt.ylabel('Coefficient Value')
plt.title('Zernike Coefficients')
plt.show()

# Remove the terms
# coefficients[0] = 0
# coefficients[1] = 0
# coefficients[2] = 0
# coefficients[3] = 0

# Reconstruct the surface using the fitted Zernike coefficients
reconstructed_surface = phaseFromZernikes(coefficients, N)


# Plot the surface and the reconstructed surface
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(surface, cmap='jet')
clb = plt.colorbar()
clb.ax.set_title('mm')
plt.title('Surface')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_surface, cmap='jet')
clb = plt.colorbar()
clb.ax.set_title('mm')
plt.title('Reconstructed Surface')

plt.show()

