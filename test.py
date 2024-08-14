from aotools.functions.zernike import zernikeArray, phaseFromZernikes, makegammas
import numpy as np
import matplotlib.pyplot as plt
import json
from zernpy import ZernPol
from math import factorial
import cv2

TEST = 5
# Define the paths
measurement_data_path = f'Measurement data/Test {TEST}/'
calibration_data_path = f'Calibration data/Test {TEST}/'
params_path = f'Results/Test {TEST}/params.json'
results_path = f'Results/Test {TEST}/'

# Load parameters from JSON file
with open(params_path, "r") as file:
    json_data = json.load(file)
scamerapx = json_data['scamerapx'] * 1e3

sscreenpx = json_data['sscreenpx']
pxperfrng = json_data['pxperfrng']

zerophase_x = json_data['zero_phase_x']
zerophase_y = json_data['zero_phase_y']

Omx = json_data['Omx']
Omy = json_data['Omy']

mask_radius = json_data['mask_radius']
mask_center_x = json_data['mask_center_x']
mask_center_y = json_data['mask_center_y']

tm_vec = json_data['mirror_tvec']
ts_vec = json_data['screen_tvec']

# Load the slope arrays
slope_x = np.load(results_path+"slope_x.npy")
slope_y = np.load(results_path+"slope_y.npy")

def create_mask(img_w, img_h, radius, center_x, center_y):
    mask = np.zeros((img_h, img_w), dtype=np.bool)
    for i in range(img_h):
        for j in range(img_w):
            if (i - center_y)**2 + (j - center_x)**2 <= radius**2:
                mask[i, j] = True
    return mask

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

# Create the mask
h, w = slope_x.shape
mask = create_mask(w, h, mask_radius, mask_center_x, mask_center_y)

# Apply the mask to the slope arrays
slope_x_whole = np.ma.array(slope_x, mask=~mask)
slope_y_whole = np.ma.array(slope_y, mask=~mask)
slope_x_masked = crop_to_unmasked(slope_x_whole)
slope_y_masked = crop_to_unmasked(slope_y_whole)

slope_x = np.ma.filled(slope_x_masked, np.nan)
slope_y = np.ma.filled(slope_y_masked, np.nan)


# Create zernike modes first derivatives maually
N = slope_x.shape[0]
# Step 1: Create a grid of x and y coordinates
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
x, y = np.meshgrid(x, y)

# Step 2: Convert (x, y) to polar coordinates (rho, theta)
rho = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)


# Step 3: Compute the gradient of the Zernike polynomial
dz1_dx, dz1_dy = np.zeros_like(slope_x), np.zeros_like(slope_x)
dz2_dx, dz2_dy = np.zeros_like(slope_x), np.ones_like(slope_x)*2
dz3_dx, dz3_dy = np.ones_like(slope_x)*2, np.zeros_like(slope_x)
dz4_dx, dz4_dy = 2*np.sqrt(6)*rho*np.sin(theta), 2*np.sqrt(6)*rho*np.cos(theta)
dz5_dx, dz5_dy = 4*np.sqrt(3)*rho*np.cos(theta), 4*np.sqrt(3)*rho*np.sin(theta)
dz6_dx, dz6_dy = 2*np.sqrt(6)*rho*np.cos(theta), -2*np.sqrt(6)*rho*np.sin(theta)
dz7_dx, dz7_dy = 3*np.sqrt(8)*rho**2*np.sin(2*theta), 3*np.sqrt(8)*rho**2**np.cos(2*theta)
dz8_dx, dz8_dy = 3*np.sqrt(8)*rho**2*np.sin(2*theta), np.sqrt(8)*((6*rho**2 - 2) - (3*rho**2*(np.cos(2*theta))**2))
dz9_dx, dz9_dy = np.sqrt(8)*((6*rho**2 - 2) + (3*rho**2*(np.cos(2*theta))**2)), 3*np.sqrt(8)*rho**2*np.sin(2*theta)
dz10_dx, dz10_dy = 3*np.sqrt(8)*rho**2*np.cos(2*theta), -3*np.sqrt(8)*rho**2*np.sin(2*theta)

# Step 4: create the Zernike gradient arrays
dz_dx = [dz1_dx, dz2_dx, dz3_dx, dz4_dx, dz5_dx, dz6_dx, dz7_dx, dz8_dx, dz9_dx, dz10_dx]
dz_dy = [dz1_dy, dz2_dy, dz3_dy, dz4_dy, dz5_dy, dz6_dy, dz7_dy, dz8_dy, dz9_dy, dz10_dy]

dz_dx_masked = [np.ma.array(z, mask=np.isnan(slope_x)) for z in dz_dx]
dz_dy_masked = [np.ma.array(z, mask=np.isnan(slope_x)) for z in dz_dy]

# Step 5: Flatten the arrays
dz_dx_flat = [z.flatten() for z in dz_dx_masked]
dz_dy_flat = [z.flatten() for z in dz_dy_masked]

slope_x_flat = slope_x_masked.flatten()
slope_y_flat = slope_y_masked.flatten()

# Step 6: Create the Slope matrix
S = np.hstack((slope_x_flat, slope_y_flat)).T
print(f"S shape = {S.shape}")

# Step 7: Create the Zernike matrix
A_list = []
for i in range(len(dz_dx_flat)):
    A_i = np.hstack((dz_dx_flat[i], dz_dy_flat[i])).T
    A_list.append(A_i)

A = np.vstack(A_list).T
print(f"A shape = {A.shape}")

# Step 8: Solve the linear system
coefficients, _, _, _ = np.linalg.lstsq(A, S, rcond=None)
print(coefficients)

# Reconstruct the surface using the fitted Zernike coefficients
reconstructed_surface = phaseFromZernikes(coefficients, N)

# Derive the reconstructed surface to compare with the slopes
slope_x_reconstructed = np.zeros_like(slope_x)
slope_y_reconstructed = np.zeros_like(slope_x)

# for i in range(len(dz_dx_masked)):
#     slope_x_reconstructed += coefficients[i]*dz_dx_masked[i]
#     slope_y_reconstructed += coefficients[i]*dz_dy_masked[i]

rcd = dz_dx_masked[1]*coefficients[1] + dz_dx_masked[2]*coefficients[2] + dz_dx_masked[3]*coefficients[3] + dz_dx_masked[4]*coefficients[4] + dz_dx_masked[5]*coefficients[5] + dz_dx_masked[6]*coefficients[6] + dz_dx_masked[7]*coefficients[7] + dz_dx_masked[8]*coefficients[8] + dz_dx_masked[9]*coefficients[9]
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(dz_dx_masked[4], cmap='jet')
plt.title('Slope X')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(slope_x, cmap='jet')
plt.title('Slope Y')
plt.colorbar()

plt.show()

slope_x_error = (slope_x_masked - slope_x_reconstructed)
slope_y_error = (slope_y_masked - slope_y_reconstructed)

# Plot the slopes
# plt.figure()

# plt.imshow(reconstructed_surface, cmap='jet')
# plt.title('Surface')
# plt.colorbar()

# plt.show()

# Plot the slopes
# plt.figure()
# plt.subplot(2, 2, 1)
# plt.imshow(slope_x_masked, cmap='jet')
# plt.title('Slope X')
# plt.colorbar()

# plt.subplot(2, 2, 2)
# plt.imshow(slope_y_masked, cmap='jet')
# plt.title('Slope Y')
# plt.colorbar()

# plt.subplot(2, 2, 3)
# plt.imshow(slope_x_reconstructed, cmap='jet')
# plt.title('Slope X Error')
# plt.colorbar()

# plt.subplot(2, 2, 4)
# plt.imshow(slope_y_reconstructed, cmap='jet')
# plt.title('Slope Y Error')
# plt.colorbar()

# plt.show()
