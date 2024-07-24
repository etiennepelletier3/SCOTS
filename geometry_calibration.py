import numpy as np
import cv2
import os
import json
import scipy
ARUCO_DICT = cv2.aruco.DICT_4X4_50  # dictionary ID

def create_aruco_marker(size, margin, flip=False):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, size, borderBits=1)
    marker = cv2.copyMakeBorder(marker, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    if flip:
        marker = cv2.flip(marker, 1)
    cv2.imshow("ArUco Marker", marker)
    cv2.waitKey(0)
    if cv2.waitKey(0) & 0xFF == ord('s'):
        # cv2.imwrite('aruco_marker.png', marker)
        cv2.destroyAllWindows()
    else:
        quit()

def aruco_pose(img, markerLength, camera_matrix, dist_coeffs):
    undist_img = cv2.undistort(img, camera_matrix, dist_coeffs)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()

    corners, ids, rejected = cv2.aruco.detectMarkers(undist_img, dictionary, parameters=params)
    
    if len(corners) > 0:
        for i in range(0, len(ids)):
            rvec, tvec, object = cv2.aruco.estimatePoseSingleMarkers(corners[i], markerLength, camera_matrix, dist_coeffs)
            print(f'Rotation Vector: {rvec}')
            print(f'Translation Vector: {tvec}')
            print(f'Object: {object}')
            # cv2.aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 255))
        cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, length=0.01, thickness=8)
    return img, rvec, tvec

def poseEstimation(img, marker_size, camera_matrix, dist_coeffs):
    undist_img = cv2.undistort(img, camera_matrix, dist_coeffs)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    # params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    corners, ids, rejected = cv2.aruco.detectMarkers(undist_img, dictionary, parameters=params)

    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    
    if len(corners) > 0:
        for corner in corners:
            _, rvec, tvec = cv2.solvePnP(marker_points, corner, camera_matrix, dist_coeffs, None, None, False, cv2.SOLVEPNP_IPPE_SQUARE)
            # Calculate the distance between the corner points in the image (in pixels)
            width = np.linalg.norm(corner[0][0] - corner[0][1])
            height = np.linalg.norm(corner[0][1] - corner[0][2])
            avg_pixel_length = (width + height) / 2

            origin_point_3d = np.array([[0, 0, 0]], dtype='float32')  # The origin in the world coordinate system
            origin_point_2d, _ = cv2.projectPoints(origin_point_3d, rvec, tvec, camera_matrix, dist_coeffs) # Project the origin point to the image frame
            center = (origin_point_2d[0][0][0], origin_point_2d[0][0][1]) # Convert to integer tuple4
            center_int = (int(center[0]), int(center[1]))
            # cv2.circle(undist_img, center_int, 10, (0, 0, 255), -1)
            cv2.aruco.drawDetectedMarkers(undist_img, corners, borderColor=(0, 255, 0))
            cv2.drawFrameAxes(undist_img, camera_matrix, dist_coeffs, rvec, tvec, length=0.01, thickness=5)

    return undist_img, center, rvec, tvec, avg_pixel_length

def screen_pose(pose_imgs_path, marker_size, camera_matrix, dist_coeffs):
    rv_mtxs = []
    tv_vecs = []
    pose_imgs = [f for f in os.listdir(pose_imgs_path) if f.startswith('pose')]
    pose_imgs.sort()
    for img in pose_imgs:
        img = cv2.imread(pose_imgs_path + img)
        img, _, rvec, tv_vec, _ = poseEstimation(img, marker_size, camera_matrix, dist_coeffs)
        rv_mtx = cv2.Rodrigues(rvec)[0]
        rv_mtxs.append(rv_mtx)
        tv_vecs.append(tv_vec)
        resized_image = cv2.resize(img, (800, 600))
        cv2.imshow('Pose', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    num_poses = len(rv_mtxs)
    eigenvector_dict = {}
    normal_vectors = []
    for i in range(num_poses):
        for j in range(num_poses):
            if i != j:
                ri_rjT = np.matmul(rv_mtxs[i], rv_mtxs[j].T)
                eigenvalues, eigenvectors = np.linalg.eig(ri_rjT)
                eigenvector = eigenvectors[:, np.isclose(eigenvalues, 1)]
                eigenvector_dict[f"m_{i+1}{j+1}"] = eigenvector
    
    m13 = eigenvector_dict['m_13'].real.flatten()
    m12 = eigenvector_dict['m_12'].real.flatten()
    m21 = eigenvector_dict['m_21'].real.flatten()
    m23 = eigenvector_dict['m_23'].real.flatten()
    
    n1 = np.cross(m13, m12) / np.linalg.norm(np.cross(m13, m12))
    n2 = np.cross(m21, m23) / np.linalg.norm(np.cross(m21, m23))
    n3 = np.cross(m13, m23) / np.linalg.norm(np.cross(m13, m23))
    n1 = np.array(n1).reshape(3, 1)
    n2 = np.array(n2).reshape(3, 1)
    n3 = np.array(n3).reshape(3, 1)
    normal_vectors.append(n1)
    normal_vectors.append(n2)
    normal_vectors.append(n3)
    
    rs_mtxs = []
    for i in range(num_poses):
        rvec = rv_mtxs[i]
        n = normal_vectors[i]
        screen_rvec = np.matmul(np.eye(3) - 2 * (n @ n.T), rvec)
        rs_mtxs.append(screen_rvec)
    
    rs_tilde = sum(rs_mtxs) / len(rs_mtxs)
    rs_tildeT_rs_tilde = np.matmul(rs_tilde.T,rs_tilde)
    rs_mtx = scipy.linalg.inv(scipy.linalg.sqrtm(rs_tildeT_rs_tilde)) @ rs_tilde
    
    rs_mtx = np.array(rs_mtx)

    I = np.eye(3)

    A1 = np.hstack([(I - n1 @ n1.T), 2 * n1, np.zeros((3, 1)), np.zeros((3, 1))])
    A2 = np.hstack([(I - n2 @ n2.T), np.zeros((3, 1)), 2 * n2, np.zeros((3, 1))])
    A3 = np.hstack([(I - n3 @ n3.T), np.zeros((3, 1)), np.zeros((3, 1)), 2 * n3])

    A = np.vstack([A1, A2, A3])

    tv_vec1 = tv_vecs[0]
    tv_vec2 = tv_vecs[1]
    tv_vec3 = tv_vecs[2]

    b = np.vstack([tv_vec1, tv_vec2, tv_vec3])

    solution = np.linalg.lstsq(A, b, rcond=None)[0]

    ts_vec = solution[:3]
    d1 = solution[3]
    d2 = solution[4]
    d3 = solution[5]

    print(f"Screen Rotation matrix: {rs_mtx}")
    print(f"Screen Translation vector: {ts_vec}")
    print(f"d1, d2, d3: {d1}, {d2}, {d3}")
    
    return rs_mtx, ts_vec

# Define the paths
measurement_data_path = 'Measurement data/Test 2/'
calibration_data_path = 'Calibration data/Test 2/'
camera_pose_path = 'Calibration data/Test 2/mirror_pose.png'
zero_phase_path = 'Calibration data/Test 2/zero_phase.png'
params_path = 'Results/Test 2/params.json'
results_path = 'Results/Test 2/'
intrinsics_path = 'Calibration data/EO-3112C_25mm-F1.4_params.json'

# Load calibration parameters from JSON file
with open(intrinsics_path, 'r') as file: # Read the JSON file
    json_data = json.load(file)
camera_matrix = np.array(json_data['camera_matrix']) # Load the camera matrix
dist_coeffs = np.array(json_data['dist_coeffs']) # Load the distortion coefficients

# Load setup parameters from JSON file
with open(params_path, 'r') as file:
    json_data = json.load(file)
pat_w = json_data['pattern_width']
pat_h = json_data['pattern_height']
sscreenpx = json_data['sscreenpx'] # Screen pixel pitch (mm per pixel)

margin = 225
marker_side = pat_w - margin*2
print(f"Marker side: {marker_side}")
marker_size = (marker_side * sscreenpx) / 1000
printed_marker_size = 0.04969 # 49.69 mm

# Find screen pose
print("...")
print("Screen pose measurement started.")
print("Take 3 different poses of the mirror and press 's'")
create_aruco_marker(marker_side, margin, flip=True)
rs_mtx, ts_vec = screen_pose(calibration_data_path, marker_size, camera_matrix, dist_coeffs)
print("Screen pose measurement completed")
print("...")

# Get zero phase point

print("Zero phase point detection started.")
print("Place the mirror to its permanent pose and take a picture of the zero phase point marker.")
create_aruco_marker(marker_side, margin, flip=True)
zero_phase_img = cv2.imread(zero_phase_path)
zero_phase_img, zero_phase_center, _, _, _ = poseEstimation(zero_phase_img, marker_size, camera_matrix, dist_coeffs)
zero_phase_x, zero_phase_y = zero_phase_center
resized_image = cv2.resize(zero_phase_img, (800, 600))  # Set the desired size
cv2.imshow('Zero_phase_image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Zero phase point detection completed.")
print("...")

# Get mirror center and pose

print("Mirror center and pose measurement started.")
print("Place the ArUco marker centered onto the mirror")
aruco_marker_captured = input("Has the ArUco marker been captured? (y/n): ")
if aruco_marker_captured.lower() == 'y':
    img = cv2.imread(camera_pose_path)
    img, center, rm_vec, tm_vec, marker_px_size = poseEstimation(img, printed_marker_size, camera_matrix, dist_coeffs)
    Omx, Omy = center
    scamerapx = (printed_marker_size / marker_px_size) * 1e3 # mm per pixel 

    resized_image = cv2.resize(img, (800, 600))  # Set the desired size
    cv2.imshow('Pose', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Mirror pose and center measurement completed.")
    print("Remove the ArUco marker")
else:
    print("Please capture the ArUco marker before proceeding.")

print("...")

# Save parameters to JSON params file
save_params = input("Do you want to save the parameters? (y/n): ")
if save_params.lower() == 'y':
    with open(params_path, 'r') as file:
        params = json.load(file)
    params.update({
        "screen_rmtx": rs_mtx.tolist(),
        "screen_tvec": ts_vec.tolist(),
        'mirror_rvec': rm_vec.tolist(),
        'mirror_tvec': tm_vec.tolist(),
        "scamerapx": scamerapx,
        "scamerapx": float(scamerapx.item()),
        "zero_phase_x": float(zero_phase_x.item()),
        "zero_phase_y": float(zero_phase_y.item()),
        'Omx': float(Omx.item()),
        'Omy': float(Omy.item()),
    })

    with open(params_path, 'w') as file:
        json.dump(params, file, indent=4)

print("Parameters saved to JSON file.")
print("Geometry calibration completed.")
print("...")

