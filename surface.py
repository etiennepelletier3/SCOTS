import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import diags
import matplotlib.pyplot as plt
import json


def create_mask(img_w, img_h, radius, center_x, center_y):
    mask = np.zeros((img_h, img_w), dtype=np.bool)
    for i in range(img_h):
        for j in range(img_w):
            if (i - center_y)**2 + (j - center_x)**2 <= radius**2:
                mask[i, j] = True
    return mask

def smooth_integration(p, q, mask, lambda_, z0, solver, precond):
    nrows, ncols = p.shape

    # Create the weight matrix
    lambda_matrix = np.ones((nrows, ncols)) * lambda_

    # Create the Laplacian matrix
    e = np.ones(nrows * ncols)
    Laplacian = diags([e, e, -4*e, e, e], [-ncols, -1, 0, 1, ncols], shape=(nrows * ncols, nrows * ncols))

    # Create the right-hand side vector
    rhs = np.zeros(nrows * ncols)
    for i in range(nrows):
        for j in range(ncols):
            if mask[i, j]:
                index = i * ncols + j
                rhs[index] = p[i, j] + q[i, j]

    # Solve the system using Conjugate Gradient
    if solver == 'pcg':
        if precond == 'ichol':
            M = diags(1 / Laplacian.diagonal())
        else:
            M = None
        z, _ = cg(Laplacian, rhs, M=M, atol=1e-8, maxiter=1000)
    else:
        z = np.linalg.solve(Laplacian.toarray(), rhs)

    return z.reshape((nrows, ncols))

def mumford_shah_integration(p, q, mask, lambda_, z0, mu, epsilon, maxit, tol, zinit):
    z = zinit.copy()
    nrows, ncols = p.shape
    for iteration in range(maxit):
        # Compute gradients
        grad_zx, grad_zy = np.gradient(z)

        # Update z using Mumford-Shah model
        numerator = (p + grad_zx) + (q + grad_zy)
        denominator = 1 + mu * (grad_zx**2 + grad_zy**2 + epsilon)
        z_new = numerator / denominator

        # Apply mask and boundary conditions
        z_new[~mask] = 0

        # Check for convergence
        if np.linalg.norm(z_new - z) < tol:
            break

        z = z_new.copy()

    return z

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

# Create the mask
h, w = slope_x.shape
mask = create_mask(w, h, mask_radius, mask_center_x, mask_center_y)
gradientMaskEroded = mask.copy()

# Apply the mask to the slope arrays
slope_x = np.ma.array(slope_x, mask=~mask)
slope_y = np.ma.array(slope_y, mask=~mask)

def trapezoidal_zonal_integration(x_slopes, y_slopes):
    # Assuming x_slopes and y_slopes are 2D numpy arrays of the same shape
    rows, cols = x_slopes.shape
    surface = np.zeros((rows, cols))
    
    # Integrate x_slopes along the x-axis
    for i in range(1, cols):
        surface[:, i] = surface[:, i-1] + 0.5 * (x_slopes[:, i-1] + x_slopes[:, i])
    
    # Integrate y_slopes along the y-axis
    for j in range(1, rows):
        surface[j, :] = surface[j-1, :] + 0.5 * (y_slopes[j-1, :] + y_slopes[j, :])
    
    return surface

def southwell_algorithm(x_slopes, y_slopes, mask, iterations=1000, tolerance=1e-6):
    rows, cols = x_slopes.shape
    surface = np.ma.array(np.zeros((rows, cols)), mask=mask)
    iteration = 0
    for _ in range(iterations):
        print(iteration)
        max_delta = 0
        
        # Update surface heights based on x_slopes
        for i in range(rows):
            for j in range(1, cols):
                if not mask[i, j] and not mask[i, j-1]:
                    delta = x_slopes[i, j-1] - (surface[i, j] - surface[i, j-1])
                    surface[i, j] += delta
                    max_delta = max(max_delta, abs(delta))
        
        # Update surface heights based on y_slopes
        for i in range(1, rows):
            for j in range(cols):
                if not mask[i, j] and not mask[i-1, j]:
                    delta = y_slopes[i-1, j] - (surface[i, j] - surface[i-1, j])
                    surface[i, j] += delta
                    max_delta = max(max_delta, abs(delta))
        iteration += 1
        if max_delta < tolerance:
            break
        
    return surface


surface_trap = trapezoidal_zonal_integration(slope_x, slope_y)
surface_south = southwell_algorithm(slope_x, slope_y, mask)
lambda_ = 1e-6
z0 = np.zeros_like(slope_x)
solver = 'pcg'
precond = 'ichol'

p = -slope_y
q = slope_x
p[np.isnan(p)] = 0
q[np.isnan(q)] = 0

print('Doing quadratic integration')

surface_quadratic = smooth_integration(p, q, gradientMaskEroded, lambda_, z0, solver, precond)

print('Doing Mumford-Shah integration')

zinit = surface_quadratic
zinit[np.isnan(zinit)] = 0
mu = 45
epsilon = 0.001
tol = 1e-8
maxit = 1000

surface_mumf = mumford_shah_integration(p, q, gradientMaskEroded, lambda_, z0, mu, epsilon, maxit, tol, zinit)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(surface_trap, cmap='jet')
clb = plt.colorbar()
clb.ax.set_title('mm')
plt.title('Surface w/ trap')

plt.subplot(1, 3, 2)
plt.imshow(surface_south, cmap='jet')
clb = plt.colorbar()
clb.ax.set_title('mm')
plt.title('Surface w/ Southwell')

plt.subplot(1, 3, 3)
plt.imshow(surface_mumf, cmap='jet')
clb = plt.colorbar()
clb.ax.set_title('mm')
plt.title('Surface w/ Mumford-Shah')

plt.show()

# Save the surface
np.save(results_path+"surface.npy", surface_mumf.data)



# def calculate_surface(xm0, ym0, zm0, xs0, ys0, zs0):
#      # Create the mirror matrix
#     m_mtx = mirror_mtx(uwrp_phase_x, xm0, ym0, zm0, Omx, Omy, scamerapx)

#     # Create the screen matrix
#     s_mtx = screen_mtx(uwrp_phase_x, uwrp_phase_y, xs0, ys0, zs0, pxperfrng, sscreenpx)

#     # Calculate the ray vectors
#     xc, yc, zc = 0, 0, 0
#     m2c, m2s = ray_vec(m_mtx, s_mtx, xc, yc, zc)

#     # Calculate the surface normal vectors
#     n_mtx = surface_normal(m2c, m2s)

#     # Calculate the surface slopes
#     slope_x, slope_y = surface_slope(n_mtx)

#     # Apply the mask to the slope arrays
#     slope_x = np.ma.array(slope_x, mask=~mask)
#     slope_y = np.ma.array(slope_y, mask=~mask)

#     int_slope_x = np.cumsum(slope_x, axis=1)
#     int_slope_y = np.cumsum(slope_y, axis=0)

#     surface = int_slope_x + int_slope_y

#     return surface

# def peak_to_peak_val(params):
#     xm0, ym0, zm0, xs0, ys0, zs0 = params
#     surface = calculate_surface(xm0, ym0, zm0, xs0, ys0, zs0)
#     print(np.ptp(surface))
#     return np.ptp(surface)


# def optimize_parameters():
#     initial_guesses = [xm0, ym0, zm0, xs0, ys0, zs0]  # Initial guesses for xm0, ym0, zm0, xs0, ys0, zs0
#     result = minimize(peak_to_peak_val, initial_guesses, options={'maxiter': 100})
#     return result.x  # Optimized values of the parameters

# # optimized_params = optimize_parameters()
# # print(f"Optimized Parameters: {optimized_params}")