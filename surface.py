import numpy as np
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.optimize import curve_fit


def mirror_mtx(mtx, xm0, ym0, zm0, Omx, Omy, scamerapx):
    xm_mtx = np.zeros_like(mtx)
    ym_mtx = np.zeros_like(mtx)
    zm_mtx = np.zeros_like(mtx)
    for j in range(mtx.shape[1]):
        xm_mtx[:, j] = xm0 - scamerapx*(j-Omx)
    for i in range(mtx.shape[0]):
        ym_mtx[i, :] = ym0 - scamerapx*(i-Omy)
    zm_mtx += zm0

    m_mtx = np.stack((xm_mtx, ym_mtx, zm_mtx), axis=0)
    return m_mtx

def screen_mtx(uwrp_phase_x, uwrp_phase_y, xs0, ys0, zs0, pxperfrng, sscreenpx):
    xs_mtx = np.zeros_like(uwrp_phase_x)
    ys_mtx = np.zeros_like(uwrp_phase_y)
    zs_mtx = np.zeros_like(uwrp_phase_x)

    xs_mtx = xs0 + 2*np.pi*uwrp_phase_x*pxperfrng*sscreenpx
    ys_mtx = ys0 + 2*np.pi*uwrp_phase_y*pxperfrng*sscreenpx
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


# Function to compute the radial Zernike polynomial
def zernike_radial(n, m, rho):
    if (n - m) % 2:
        return np.zeros_like(rho)
    radial_poly = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        radial_poly += rho ** (n - 2 * k) * (-1) ** k * factorial(n - k) / (
            factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        )
    return radial_poly

# Function to compute the full Zernike polynomial
def zernike_polynomial(n, m, rho, theta):
    if m >= 0:
        return zernike_radial(n, m, rho) * np.cos(m * theta)
    else:
        return zernike_radial(n, -m, rho) * np.sin(-m * theta)

# Function to fit Zernike coefficients
def fit_zernike_coefficients(height_map, mask, N):
    y, x = np.indices(height_map.shape)
    y = y - np.mean(y)
    x = x - np.mean(x)
    rho = np.sqrt(x**2 + y**2) / np.max(np.sqrt(x**2 + y**2))
    theta = np.arctan2(y, x)

    valid_mask = (rho <= 1) & (~mask)
    rho = rho[valid_mask]
    theta = theta[valid_mask]
    height_values = height_map[valid_mask]

    def zernike_sum(rho_theta, *coeffs):
        rho, theta = rho_theta
        result = np.zeros_like(rho)
        index = 0
        for n in range(N + 1):
            for m in range(-n, n + 1, 2):
                result += coeffs[index] * zernike_polynomial(n, m, rho, theta)
                index += 1
        return result

    initial_guess = np.zeros((N + 1) * (N + 2) // 2)
    rho_theta = np.vstack((rho, theta))
    coeffs, _ = curve_fit(zernike_sum, rho_theta, height_values, p0=initial_guess)

    return coeffs

# Function to reconstruct the surface from Zernike coefficients
def reconstruct_surface(coeffs, N, grid_size):
    y, x = np.indices((grid_size, grid_size))
    y = y - np.mean(y)
    x = x - np.mean(x)
    rho = np.sqrt(x**2 + y**2) / np.max(np.sqrt(x**2 + y**2))
    theta = np.arctan2(y, x)

    reconstructed = np.zeros_like(rho)
    index = 0
    for n in range(N + 1):
        for m in range(-n, n + 1, 2):
            reconstructed += coeffs[index] * zernike_polynomial(n, m, rho, theta)
            index += 1

    reconstructed[rho > 1] = np.nan  # Mask out values outside the unit disk
    return reconstructed


# Define the paths
measurement_data_path = 'Measurement data/Test 2/'
calibration_data_path = 'Calibration data/Test 2/'
params_path = 'Results/Test 2/params.json'
results_path = 'Results/Test 2/'

# Load parameters from JSON file
with open(params_path, "r") as file:
    json_data = json.load(file)
scamerapx = json_data['scamerapx']

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

# Load unwrapped phases
uwrp_phase_x = np.load(results_path + "uwrp_phase_x.npy")
uwrp_phase_y = np.load(results_path + "uwrp_phase_y.npy")

# Create the mask
h, w = uwrp_phase_x.shape
mask = create_mask(w, h, mask_radius, mask_center_x, mask_center_y)

# Create a masked array
uwrp_phase_x = np.ma.array(uwrp_phase_x, mask=~mask)
uwrp_phase_y = np.ma.array(uwrp_phase_y, mask=~mask)

# Create the mirror matrix
xm0 = tm_vec[0][0]
ym0 = tm_vec[1][0]
zm0 = tm_vec[2][0]
m_mtx = mirror_mtx(uwrp_phase_x, xm0, ym0, zm0, Omx, Omy, scamerapx)

# Create the screen matrix
xs0 = ts_vec[0][0]
ys0 = ts_vec[1][0]
zs0 = ts_vec[2][0]
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

# Save the slopes
np.save(results_path+"slope_x.npy", slope_x.data)
np.save(results_path+"slope_y.npy", slope_y.data)

# Integrate the slopes
int_slope_x = np.cumsum(slope_x, axis=1)
int_slope_y = np.cumsum(slope_y, axis=0)

surface = int_slope_x + int_slope_y

# Plot the surface
plt.imshow(surface, cmap='jet')
clb = plt.colorbar()
clb.ax.set_title('mm')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Save the surface
np.save(results_path+"surface.npy", surface.data)

N = 3  # Maximum radial degree
coeffs = fit_zernike_coefficients(surface.data, surface.mask, N)
reconstructed_surface = reconstruct_surface(coeffs, N, surface.shape[0])

print("Zernike coefficients:")
print(coeffs)

# Plot the reconstructed surface
plt.imshow(reconstructed_surface, cmap='jet')
plt.colorbar(label='mm')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()



