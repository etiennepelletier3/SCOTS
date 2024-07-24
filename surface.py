import numpy as np
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def mirror_mtx(mtx, xm0, ym0, zm0, Omx, Omy, scamerapx):
    xm_mtx = np.zeros_like(mtx)
    ym_mtx = np.zeros_like(mtx)
    zm_mtx = np.zeros_like(mtx)
    for j in range(mtx.shape[1]):
        xm_mtx[:, j] = xm0 + scamerapx*(j-Omx)
    for i in range(mtx.shape[0]):
        ym_mtx[i, :] = ym0 - scamerapx*(i-Omy)
    zm_mtx += zm0

    m_mtx = np.stack((xm_mtx, ym_mtx, zm_mtx), axis=0)
    return m_mtx

def screen_mtx(uwrp_phase_x, uwrp_phase_y, xs0, ys0, zs0, pxperfrng, sscreenpx):
    xs_mtx = np.zeros_like(uwrp_phase_x)
    ys_mtx = np.zeros_like(uwrp_phase_y)
    zs_mtx = np.zeros_like(uwrp_phase_x)

    xs_mtx = xs0 + uwrp_phase_x*pxperfrng*sscreenpx/(2*np.pi)
    ys_mtx = ys0 + uwrp_phase_y*pxperfrng*sscreenpx/(2*np.pi)
    zs_mtx += zs0
    
    s_mtx = np.stack((xs_mtx, ys_mtx, zs_mtx), axis=0)
    return s_mtx

def ray_vec(m_mtx, s_mtx, xc, yc, zc):
    m2c_x = xc - m_mtx[0, :, :]
    m2c_y = yc - m_mtx[1, :, :]
    m2c_z = zc - m_mtx[2, :, :]

    m2c = np.stack((m2c_x, m2c_y, m2c_z))

    m2s_x = s_mtx[0, :, :] - m_mtx[0, :, :]
    m2s_y = s_mtx[1, :, :] - m_mtx[1, :, :]
    m2s_z = s_mtx[2, :, :] - m_mtx[2, :, :]

    m2s = np.stack((m2s_x, m2s_y, m2s_z))

    return m2c, m2s

def surface_normal(m2c, m2s):
    # Calculate surface vector normal to the mirror
    m2c_norm = np.linalg.norm(m2c, axis=0)
    m2s_norm = np.linalg.norm(m2s, axis=0)

    N_mtx = m2c / m2c_norm + m2s / m2s_norm

    n_mtx = N_mtx / np.linalg.norm(N_mtx, axis=0)
    return n_mtx

def surface_slope(n_mtx):
    # Calculate surface slope
    slope_x = -n_mtx[0, :, :] / n_mtx[2, :, :]
    slope_y = -n_mtx[1, :, :] / n_mtx[2, :, :]
    return slope_x, slope_y

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

# Define the paths
measurement_data_path = 'Measurement data/Test 2/'
calibration_data_path = 'Calibration data/Test 2/'
params_path = 'Results/Test 2/params.json'
results_path = 'Results/Test 2/'

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

manual_geometry = False

# Load unwrapped phases
uwrp_phase_x = np.load(results_path + "uwrp_phase_x.npy")
uwrp_phase_y = np.load(results_path + "uwrp_phase_y.npy")

# Create the mask
h, w = uwrp_phase_x.shape
mask = create_mask(w, h, mask_radius, mask_center_x, mask_center_y)

# Create a masked array
uwrp_phase_x = np.ma.array(uwrp_phase_x, mask=~mask)
uwrp_phase_y = np.ma.array(uwrp_phase_y, mask=~mask)

if manual_geometry:
    # Define the parameters
    xm0 = 0.0010983523035546735
    ym0 = -0.006166640536640525
    zm0 = 0.6446248975662893
    xs0 = 0.6516130693249402
    ys0 = 0.042220644356549965
    zs0 = 0.001

else:
    xm0 = tm_vec[0][0] * 1e3 # Convert to mm
    ym0 = tm_vec[1][0] * 1e3 # Convert to mm
    zm0 = tm_vec[2][0] * 1e3 # Convert to mm

    xs0 = ts_vec[0][0] * 1e3 # Convert to mm
    ys0 = ts_vec[1][0] * 1e3 # Convert to mm
    zs0 = ts_vec[2][0] * 1e3 # Convert to mm

# Create the mirror matrix
m_mtx = mirror_mtx(uwrp_phase_x, xm0, ym0, zm0, Omx, Omy, scamerapx)

# Create the screen matrix
s_mtx = screen_mtx(uwrp_phase_x, uwrp_phase_y, xs0, ys0, zs0, pxperfrng, sscreenpx)
# plt.figure()
# plt.imshow(s_mtx[0,:, :], cmap='jet')
# plt.colorbar()
# plt.title('Screen Matrix')
# Calculate the ray vectors
xc, yc, zc = 0, 0, 0
m2c, m2s = ray_vec(m_mtx, s_mtx, xc, yc, zc)

# Calculate the surface normal vectors
n_mtx = surface_normal(m2c, m2s)

# Calculate the surface slopes
slope_x, slope_y = surface_slope(n_mtx)


# Apply the mask to the slope arrays
slope_x = np.ma.array(slope_x, mask=~mask)
slope_y = np.ma.array(slope_y, mask=~mask)

# Plot the x and y slope
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(slope_x, cmap='jet')
plt.colorbar()
plt.title('X Slope')
plt.subplot(1, 2, 2)
plt.imshow(slope_y, cmap='jet')
plt.colorbar()
plt.title('Y Slope')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(14, 6))

# Define X and Y
X = np.arange(slope_x.shape[1])
Y = np.arange(slope_x.shape[0])
X, Y = np.meshgrid(X, Y)

# First subplot for Z1
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, slope_x, cmap='viridis')
ax1.set_title('X Slope')
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')

# Second subplot for Z2
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, slope_y, cmap='plasma')
ax2.set_title('Y Slope')
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('Z axis')

plt.show()



# Save the slopes
np.save(results_path+"slope_x.npy", slope_x.data)
np.save(results_path+"slope_y.npy", slope_y.data)


# Integrate the slopes
int_slope_x = np.cumsum(slope_x, axis=1)
int_slope_y = np.cumsum(slope_y, axis=0)

surface = int_slope_x + int_slope_y


# surface_southwell = southwell_method(slope_x, slope_y, scamerapx, scamerapx)

# Plot the surface
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(surface, cmap='jet')
plt.colorbar()
plt.title('Surface')
# plt.subplot(2, 2, 2)
# plt.imshow(surface_southwell, cmap='jet')
# plt.colorbar()
# plt.title('Surface Southwell')


plt.tight_layout()
plt.show()


# Save the surface
np.save(results_path+"surface.npy", surface.data)

def calculate_surface(xm0, ym0, zm0, xs0, ys0, zs0):
     # Create the mirror matrix
    m_mtx = mirror_mtx(uwrp_phase_x, xm0, ym0, zm0, Omx, Omy, scamerapx)

    # Create the screen matrix
    s_mtx = screen_mtx(uwrp_phase_x, uwrp_phase_y, xs0, ys0, zs0, pxperfrng, sscreenpx)

    # Calculate the ray vectors
    xc, yc, zc = 0, 0, 0
    m2c, m2s = ray_vec(m_mtx, s_mtx, xc, yc, zc)

    # Calculate the surface normal vectors
    n_mtx = surface_normal(m2c, m2s)

    # Calculate the surface slopes
    slope_x, slope_y = surface_slope(n_mtx)

    # Apply the mask to the slope arrays
    slope_x = np.ma.array(slope_x, mask=~mask)
    slope_y = np.ma.array(slope_y, mask=~mask)

    int_slope_x = np.cumsum(slope_x, axis=1)
    int_slope_y = np.cumsum(slope_y, axis=0)

    surface = int_slope_x + int_slope_y

    return surface

def peak_to_peak_val(params):
    xm0, ym0, zm0, xs0, ys0, zs0 = params
    surface = calculate_surface(xm0, ym0, zm0, xs0, ys0, zs0)
    print(np.ptp(surface))
    return np.ptp(surface)


def optimize_parameters():
    initial_guesses = [xm0, ym0, zm0, xs0, ys0, zs0]  # Initial guesses for xm0, ym0, zm0, xs0, ys0, zs0
    result = minimize(peak_to_peak_val, initial_guesses, options={'maxiter': 100})
    return result.x  # Optimized values of the parameters

# optimized_params = optimize_parameters()
# print(f"Optimized Parameters: {optimized_params}")