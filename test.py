from aotools.functions.zernike import zernikeArray, phaseFromZernikes
import numpy as np
import matplotlib.pyplot as plt
import json

TEST = 3
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
slope_x = np.ma.array(slope_x, mask=~mask)
slope_y = np.ma.array(slope_y, mask=~mask)
slope_x = crop_to_unmasked(slope_x)
slope_y = crop_to_unmasked(slope_y)
slope_x = np.ma.filled(slope_x, np.nan)
slope_y = np.ma.filled(slope_y, np.nan)


def southwell_algorithm(x_slopes, y_slopes, iterations=1000, tolerance=1e-6):
    rows, cols = x_slopes.shape
    surface = np.zeros((rows, cols))
    
    it=0
    for _ in range(iterations):
        print(it)
        max_delta = 0
        
        # Update surface heights based on x_slopes
        for i in range(rows):
            for j in range(1, cols):
                if not np.isnan(x_slopes[i, j-1]):
                    delta = x_slopes[i, j-1] - (surface[i, j] - surface[i, j-1])
                    surface[i, j] += delta
                    max_delta = max(max_delta, abs(delta))
        
        # Update surface heights based on y_slopes
        for i in range(1, rows):
            for j in range(cols):
                if not np.isnan(y_slopes[i-1, j]):
                    delta = y_slopes[i-1, j] - (surface[i, j] - surface[i-1, j])
                    surface[i, j] += delta
                    max_delta = max(max_delta, abs(delta))
        
        if max_delta < tolerance:
            break
        
        it += 1
    return surface


def zernike_fit_slopes(x_slopes, y_slopes, zernike_order):
    rows, cols = x_slopes.shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    rho, theta = np.sqrt(X**2 + Y**2), np.arctan2(Y, X)
    
    # Mask for the unit circle
    mask = rho <= 1
    
    # Get Zernike modes
    zernike_modes = zernikeArray(zernike_order, rows)
    
    # Flatten the arrays and mask
    x_slopes_flat = x_slopes[mask]
    y_slopes_flat = y_slopes[mask]
    zernike_modes_flat = zernike_modes[:, mask]
    
    # Construct design matrix for x and y slopes
    A_x = np.zeros((x_slopes_flat.size, zernike_modes.shape[0]))
    A_y = np.zeros((y_slopes_flat.size, zernike_modes.shape[0]))
    for i in range(zernike_modes.shape[0]):
        A_x[:, i] = zernike_modes_flat[i] * X[mask]
        A_y[:, i] = zernike_modes_flat[i] * Y[mask]
    
    # Concatenate the design matrices and slope arrays
    A = np.vstack((A_x, A_y))
    slopes = np.concatenate((x_slopes_flat, y_slopes_flat))
    
    # Solve for Zernike coefficients
    zernike_coefficients, _, _, _ = np.linalg.lstsq(A, slopes, rcond=None)
    
    # Reconstruct the surface from the Zernike coefficients
    surface = np.zeros_like(X)
    for i in range(zernike_modes.shape[0]):
        surface += zernike_coefficients[i] * zernike_modes[i]
    
    surface[~mask] = np.nan  # Mask out the outside region
    
    return surface

# surface_south = southwell_algorithm(slope_x, slope_y, 100, 1e-6)

surface_zern = zernike_fit_slopes(slope_x, slope_y, 20)

# Plot the slopes
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(slope_y, cmap='jet')
plt.title('Slope X')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(surface_zern, cmap='jet')
plt.title('Slope Y')
plt.colorbar()

plt.show()