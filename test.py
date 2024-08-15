from aotools.functions.zernike import zernikeArray, phaseFromZernikes
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

# Remove tip from the initial slopes
slope_x -= np.mean(slope_x_masked)
slope_y -= np.mean(slope_y_masked)


# Create zernike modes first derivatives maually
N = slope_x.shape[0]
# Step 1: Create a grid of x and y coordinates
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
x, y = np.meshgrid(x, y)

# Step 3: Compute the gradient of the Zernike polynomial (noll ordering)
dz0_dx, dz0_dy = np.zeros_like(slope_x), np.zeros_like(slope_x)
dz1_dx, dz1_dy = np.zeros_like(slope_x), np.ones_like(slope_x)*2
dz2_dx, dz2_dy = np.ones_like(slope_x)*2, np.zeros_like(slope_x)
dz3_dx, dz3_dy = 2*np.sqrt(6)*y, 2*np.sqrt(6)*x
dz4_dx, dz4_dy = 4*np.sqrt(3)*x, 4*np.sqrt(3)*y
dz5_dx, dz5_dy = 2*np.sqrt(6)*x, -2*np.sqrt(6)*y
dz6_dx, dz6_dy = 6*np.sqrt(8)*x*y, 3*np.sqrt(8)*(x**2 - y**2)
dz7_dx, dz7_dy = 6*np.sqrt(8)*x*y, np.sqrt(8)*(3*x**2 + 9*y**2 - 2)
dz8_dx, dz8_dy = np.sqrt(8)*(9*x**2 + 3*y**2 - 2), 6*np.sqrt(8)*x*y
dz9_dx, dz9_dy = np.sqrt(8)*(3*x**2 - 3*y**2), -6*np.sqrt(8)*x*y
dz10_dx, dz10_dy = np.sqrt(10)*(12*x**2*y - 4*y**3), np.sqrt(10)*(4*x**3 - 12*x*y**2)
dz11_dx, dz11_dy = np.sqrt(10)*(24*x**2*y  + 8*y**3 - 6*y), np.sqrt(10)*(8*x**3 + 24*x*y**2 - 6*x)
dz12_dx, dz12_dy = np.sqrt(5)*(24*x**3 + 24*x*y**2 - 12*x), np.sqrt(5)*(24*x**2*y + 24*y**3 - 12*y)
dz13_dx, dz13_dy = np.sqrt(10)*(16*x**3 - 6*x), np.sqrt(10)*(-16*y**3 + 6*y)
dz14_dx, dz14_dy = np.sqrt(10)*(4*x**3 - 12*x*y**2), np.sqrt(10)*(-12*x**2*y + 4*y**3)
dz15_dx, dz15_dy = np.sqrt(12)*(20*x**3*y - 20*x*y**3), np.sqrt(12)*(5*x**4 - 30*x**2*y**2 + 5*y**4)
dz16_dx, dz16_dy = np.sqrt(12)*(60*x**3*y + 20*x*y**3 - 24*x*y - 10), np.sqrt(12)*(15*x**4 + 30*x**2*y**2 - 25*y**4 - 12*x**2 + 12*y**2)
dz17_dx, dz17_dy = np.sqrt(12)*(40*x**3*y + 40*x*y**3 - 24*x*y), np.sqrt(12)*(10*x**4 + 60*x**2*y**2 + 50*y**4 - 12*x**2 - 36*y**2 + 3)
dz18_dx, dz18_dy = np.sqrt(12)*(50*x**4 + 60*x**2*y**2 + 10*y**4 - 36*x**2 - 12*y**2 + 3), np.sqrt(12)*(40*x**3*y + 40*x*y**3 - 24*x*y)
dz19_dx, dz19_dy = np.sqrt(12)*(25*x**4 - 30*x**2*y**2 - 15*y**4 - 12*x**2 + 12*y**2), np.sqrt(12)*(-20*x**3*y - 60*x*y**3 + 24*x*y)
dz20_dx, dz20_dy = np.sqrt(12)*(5*x**4 - 30*x**2*y**2 + 5*y**4), np.sqrt(12)*(-20*x**3*y + 20*x*y**3)

# Step 4: create the Zernike gradient arrays
dz_dx = [dz0_dx, dz1_dx, dz2_dx, dz3_dx, dz4_dx, dz5_dx, dz6_dx, dz7_dx, dz8_dx, dz9_dx, dz10_dx, dz11_dx, dz12_dx, dz13_dx, dz14_dx, dz15_dx, dz16_dx, dz17_dx, dz18_dx, dz19_dx, dz20_dx]
dz_dy = [dz0_dy, dz1_dy, dz2_dy, dz3_dy, dz4_dy, dz5_dy, dz6_dy, dz7_dy, dz8_dy, dz9_dy, dz10_dy, dz11_dy, dz12_dy, dz13_dy, dz14_dy, dz15_dy, dz16_dy, dz17_dy, dz18_dy, dz19_dy, dz20_dy]

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
print(f"coefficients: {coefficients}")


# Reconstruct the surface using the fitted Zernike coefficients
R = np.sqrt(x**2 + y**2)
R[R > 1] = np.nan
Theta = np.arctan2(y, x)

polynomials = []
for OSA_index in range(len(coefficients)):
    zern = ZernPol(osa_index=OSA_index)
    mode = zern.polynomial_value(R, Theta, use_exact_eq=True)
    if OSA_index == 0:
        mode = np.ma.array(mode, mask=np.isnan(slope_x))
    # plt.figure()
    # plt.imshow(mode, cmap='jet')
    # plt.colorbar()
    # plt.title(f'Zernike Mode {OSA_index}')
    # plt.show()
    polynomials.append(zern)


surface = ZernPol.sum_zernikes(coefficients, polynomials, R, Theta)

plt.figure()
plt.imshow(surface, cmap='jet')
clb = plt.colorbar()
clb.ax.set_title('mm')
plt.title(f'Surface')
plt.show()

# Derive the reconstructed surface to get the slopes
grad_surface = np.gradient(surface)
slope_x_derived = grad_surface[1]
slope_y_derived = grad_surface[0]

# Reconstruct the slopes using the coefficients and the Zernike polynomials derivatives
slope_x_reconstructed = np.ma.array(np.zeros_like(slope_x), mask=np.isnan(slope_x))
slope_y_reconstructed = np.ma.array(np.zeros_like(slope_x), mask=np.isnan(slope_x))

for i in range(1, len(dz_dx_masked)):
    slope_x_reconstructed += coefficients[i]*dz_dx_masked[i]
    slope_y_reconstructed += coefficients[i]*dz_dy_masked[i]

# Calculate the error
# slope_x_error = (slope_x_masked - slope_x_reconstructed)
# slope_y_error = (slope_y_masked - slope_y_reconstructed)

# Plot the slopes
plt.figure()

plt.subplot(2, 3, 1)
plt.imshow(slope_x, cmap='jet')
plt.title('Original x slope')
clb = plt.colorbar()
clb.ax.set_title('mm')

plt.subplot(2, 3, 4)
plt.imshow(slope_y, cmap='jet')
plt.title('Original y slope')
clb = plt.colorbar()
clb.ax.set_title('mm')

plt.subplot(2, 3, 2)
plt.imshow(slope_x_reconstructed, cmap='jet')
plt.title('Reconstructed x slope')
clb = plt.colorbar()
clb.ax.set_title('mm')

plt.subplot(2, 3, 5)
plt.imshow(slope_y_reconstructed, cmap='jet')
plt.title('Reconstructed y slope')
clb = plt.colorbar()
clb.ax.set_title('mm')

plt.subplot(2, 3, 3)
plt.imshow(slope_x_derived, cmap='jet')
plt.title('Derived x slope')
clb = plt.colorbar()
clb.ax.set_title('mm')

plt.subplot(2, 3, 6)
plt.imshow(slope_y_derived, cmap='jet')
plt.title('Derived y slope')
clb = plt.colorbar()
clb.ax.set_title('mm')

plt.show()

