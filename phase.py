import numpy as np
import cv2
import os
from scipy.optimize import minimize
from skimage.restoration import unwrap_phase
import json
import matplotlib.pyplot as plt


def pattern(pat_w, pat_h, nb_shifts, pxperfrng, Osx, Osy):
    # Phase shifts
    phases = np.linspace(0, 2*np.pi, nb_shifts)

    # Create the fringe patterns
    fringe_pattern_x = np.zeros((pat_h, pat_w), dtype=np.uint8)
    fringe_pattern_y = np.zeros((pat_h, pat_w), dtype=np.uint8)

    for phase in phases:
        for x in range(pat_h):
            fringe_pattern_x[:, x] = 255 * (1 + np.sin(2*np.pi*(x-Osx)/pxperfrng + phase))/2

        for y in range(pat_w):
            fringe_pattern_y[y, :] = 255 * (1 + np.sin(2*np.pi*(y-Osy)/pxperfrng + phase))/2

        fringe_pattern_x = fringe_pattern_x.astype(np.uint8)
        fringe_pattern_y = fringe_pattern_y.astype(np.uint8)
        
        cv2.imshow('Fringe Pattern x', fringe_pattern_x)
        cv2.waitKey(0)
        cv2.imshow('Fringe Pattern y', fringe_pattern_y)
        cv2.waitKey(0)

def create_mask(img_w, img_h, radius, center_x, center_y):
    mask = np.zeros((img_h, img_w), dtype=np.bool)
    for i in range(img_h):
        for j in range(img_w):
            if (i - center_y)**2 + (j - center_x)**2 <= radius**2:
                mask[i, j] = True
    return mask

def rescale_intensity(img, min_val=-1, max_val=1):
    # Convert the image to float32 for precision
    img = img.astype(np.float32)
    
    # Get the current minimum and maximum values
    curr_min = img.min()
    curr_max = img.max()
    
    # Rescale the image intensity to the range [min_val, max_val]
    rescaled_image = (img - curr_min) / (curr_max - curr_min) * (max_val - min_val) + min_val
    
    return rescaled_image

def load_captured_patterns(measurement_data_path):
    x_fringe_files = [f for f in os.listdir(measurement_data_path) if f.startswith('X')]
    y_fringe_files = [f for f in os.listdir(measurement_data_path) if f.startswith('Y')]
    x_fringe_files.sort()
    y_fringe_files.sort()
    x_fringe_imgs = []
    y_fringe_imgs = []
    for f in x_fringe_files:
        img = cv2.imread(measurement_data_path + f, cv2.IMREAD_GRAYSCALE)
        # Undistort the image
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
        x_fringe_imgs.append(undistorted_img)
    for f in y_fringe_files:
        img = cv2.imread(measurement_data_path + f, cv2.IMREAD_GRAYSCALE)
        # Undistort the image
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
        y_fringe_imgs.append(undistorted_img)
    return x_fringe_imgs, y_fringe_imgs

def rescale_imgs_list(x_fringe_imgs, y_fringe_imgs, mask_for_img):
    rescaled_imgs_x = []
    rescaled_imgs_y = []
    for img in x_fringe_imgs:
        masked_img = cv2.bitwise_and(img, img, mask=mask_for_img)
        rescaled_img = rescale_intensity(masked_img)
        rescaled_imgs_x.append(rescaled_img)

    for img in y_fringe_imgs:
        masked_img = cv2.bitwise_and(img, img, mask=mask_for_img)
        rescaled_img = rescale_intensity(masked_img)
        rescaled_imgs_y.append(rescaled_img)

    # Stack all the images into a single array
    Ixs = np.stack(rescaled_imgs_x, axis=0)
    Iys = np.stack(rescaled_imgs_y, axis=0)

    return Ixs, Iys

def objective_function(phi, theta, intensity_values):
    return np.sum((intensity_values - np.sin(theta + phi))**2)

def calculate_wrapped_phase(mask, Ixs, Iys, wrp_phase_x, wrp_phase_y, theta, initial_guess):
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                print(f"Processing pixel ({i}, {j})")
                intensity_values_x = Ixs[:, i, j]
                result_x = minimize(objective_function, initial_guess, args=(theta, intensity_values_x))
                wrp_phase_x[i, j] = result_x.x[0]
                intensity_values_y = Iys[:, i, j]
                result_y = minimize(objective_function, initial_guess, args=(theta, intensity_values_y))
                wrp_phase_y[i, j] = result_y.x[0]

    return wrp_phase_x, wrp_phase_y

# Define the paths
measurement_data_path = 'Measurement data/Test 2/'
calibration_data_path = 'Calibration data/Test 2/'
params_path = 'Results/Test 2/params.json'
results_path = 'Results/Test 2/'
intrinsics_path = 'Calibration data/EO-3112C_25mm-F1.4_params.json'

# Load parameters from JSON file
with open(params_path, "r") as file:
    json_data = json.load(file)
pat_w = json_data['pattern_width']
pat_h = json_data['pattern_height']
nb_shifts = json_data['nb_shifts']
pxperfrng = json_data['pxperfrng']
zerophase_x = json_data['zero_phase_x']
zerophase_y = json_data['zero_phase_y']
Omx = json_data['Omx']
Omy = json_data['Omy']

# Load calibration parameters from JSON files
with open(intrinsics_path, 'r') as file: # Read the JSON file
    json_data = json.load(file)
camera_matrix = np.array(json_data['camera_matrix']) # Load the camera matrix
dist_coeffs = np.array(json_data['dist_coeffs']) # Load the distortion coefficients

# Display the fringe patterns
pattern(pat_w, pat_h, nb_shifts, pxperfrng, Omx, Omy)

# Load captured fringe patterns
x_fringe_imgs, y_fringe_imgs = load_captured_patterns(measurement_data_path)

# Get the image dimensions
img_h, img_w = x_fringe_imgs[0].shape

# Create a while loop for mask creation
create_mask_loop = True
while create_mask_loop:
    # Ask the user if they want to create a mask or not
    choice = input("Do you want to create a mask? (y/n): ")
    
    if choice.lower() == 'y':
        # Get mask parameters from user
        print("image width:", img_w)
        print("image height:", img_h)
        mask_radius = int(input("Enter the radius of the mask: "))
        mask_center_x = Omx
        mask_center_y = Omy
        
        # Create the mask
        mask = create_mask(img_w, img_h, mask_radius, mask_center_x, mask_center_y)

        # Convert the mask to 0 or 255
        mask_for_img = mask.astype(np.uint8) * 255
        
        # Apply the mask to a sample image
        sample_img = x_fringe_imgs[0]
        masked_img = cv2.bitwise_and(sample_img, sample_img, mask=mask_for_img)
        
        # Display the masked image
        resized_img = cv2.resize(masked_img, (800, 600))
        cv2.imshow('Masked Image', resized_img)
        cv2.waitKey(0)
        
        # Ask the user if they want to keep the mask or create a new one
        choice = input("Do you want to keep this mask? (y/n): ")
        
        if choice.lower() == 'y':
            # Save mask parameters to JSON file
            with open(params_path, 'r') as file:
                params = json.load(file)
            params.update({
                'mask_radius': mask_radius,
                'mask_center_x': mask_center_x,
                'mask_center_y': mask_center_y
            })

            with open(params_path, 'w') as file:
                json.dump(params, file, indent=4)
            break
    else:
        mask = np.ones((img_h, img_w), dtype=np.bool)
        mask_for_img = np.ones((img_h, img_w), dtype=np.uint8) * 255
        create_mask_loop = False
        break
    
# Rescale the intensity values of the images from -1 to 1
Ixs, Iys = rescale_imgs_list(x_fringe_imgs, y_fringe_imgs, mask_for_img)

# Define the phase shift values
theta = np.linspace(0, 2*np.pi, nb_shifts)

# Define the initial guess for the phases
initial_guess = 0
wrp_phase_x = np.zeros_like(Ixs[0, :, :])
wrp_phase_y = np.zeros_like(Iys[0, :, :])

# Calculate the wrapped phases
wrp_phase_x, wrp_phase_y = calculate_wrapped_phase(mask, Ixs, Iys, wrp_phase_x, wrp_phase_y, theta, initial_guess)

# Save the wrapped phases
np.save(results_path + "wrp_phase_x.npy", wrp_phase_x)
np.save(results_path + "wrp_phase_y.npy", wrp_phase_y)

# Apply the mask to the wrapped phases
wrp_phase_x = np.ma.array(wrp_phase_x, mask=~mask)
wrp_phase_y = np.ma.array(wrp_phase_y, mask=~mask)

# Unwrap the phases
unwrap_phase_x = unwrap_phase(wrp_phase_x)
unwrap_phase_y = unwrap_phase(wrp_phase_y)

# Apply zero phase offset
val_at_zerophase_x = unwrap_phase_x[int(zerophase_y), int(zerophase_x)]
val_at_zerophase_y = unwrap_phase_y[int(zerophase_y), int(zerophase_x)]
unwrap_phase_x -= val_at_zerophase_x
unwrap_phase_y -= val_at_zerophase_y

# Get the minimum and maximum phase values for display
min_phase_x = unwrap_phase_x[int(mask_center_y), int(mask_center_x)+mask_radius-1]
min_phase_y = unwrap_phase_y[int(mask_center_y)-mask_radius+1, int(mask_center_x)]
max_phase_x = unwrap_phase_x[int(mask_center_y), int(mask_center_x)-mask_radius+1]
max_phase_y = unwrap_phase_y[int(mask_center_y)+mask_radius-1, int(mask_center_x)]

# Display the results
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(wrp_phase_x, cmap='jet', vmin=-2*np.pi, vmax=2*np.pi)
plt.title('Wrp Phase X')
plt.subplot(2, 2, 2)
plt.imshow(wrp_phase_y, cmap='jet', vmin=-2*np.pi, vmax=2*np.pi)
plt.title('Wrp Phase Y')
plt.subplot(2, 2, 3)
plt.imshow(unwrap_phase_x, cmap='jet', vmin=min_phase_x, vmax=max_phase_x)
plt.title('Unwrap Phase X')
plt.subplot(2, 2, 4)
plt.imshow(unwrap_phase_y, cmap='jet', vmin=min_phase_y, vmax=max_phase_y)
plt.title('Unwrap Phase Y')

plt.tight_layout()
plt.show()

# Save the unwrapped phases
np.save(results_path + 'uwrp_phase_x.npy', unwrap_phase_x.data)
np.save(results_path + 'uwrp_phase_y.npy', unwrap_phase_y.data)