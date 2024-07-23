import cv2
import numpy as np
import os
import json

# Enter setup parameters
sscreenpx = 0.223
ARUCO_DICT = cv2.aruco.DICT_4X4_50  # dictionary ID
SQUARES_VERTICALLY = 5              # number of squares vertically
SQUARES_HORIZONTALLY = 4            # number of squares horizontally
LENGTH_PX = 1080                    # size of the board in pixels
MARGIN_PX = 50                      # size of the margin in pixels
SQUARE_LENGTH = (((SQUARES_HORIZONTALLY/SQUARES_VERTICALLY)*LENGTH_PX - 2*MARGIN_PX)/SQUARES_HORIZONTALLY)*sscreenpx*0.001   # square side length (m)
print(SQUARE_LENGTH)
MARKER_LENGTH = 0.025               # ArUco marker side length (m)
CALIB_DIR_PATH = 'Calibration data/' # path to the folder with images

# Create ChArUco board
def create_ChArUco_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)
    cv2.imshow("ChArUco", img)
    cv2.waitKey(0)
    if cv2.waitKey(0) & 0xFF == ord('s'):
        cv2.imwrite(CALIB_DIR_PATH + 'ChAruco_board.bmp', img)
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()

def get_intrinsic_parameters():
    # Define the aruco dictionary and charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()

    # Load PNG images from folder
    image_files = [os.path.join(CALIB_DIR_PATH, f) for f in os.listdir(CALIB_DIR_PATH) if f.startswith("intrinsic")]
    image_files.sort()  # Ensure files are in order
    print(f"Number of images found: {len(image_files)}")

    all_charuco_corners = []
    all_charuco_ids = []

    for image_file in image_files:
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        image_copy = image.copy()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)
        
        # If at least one marker is detected
        if len(marker_ids) > 0:
            # cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
            # resized_image_copy = cv2.resize(image_copy, (800, 800))  # Set the desired size
            # cv2.imshow('Detected Markers', resized_image_copy)
            # cv2.waitKey(0)
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)
            if charuco_retval and len(charuco_corners) > 3:
                all_charuco_corners.append(charuco_corners)
                
                all_charuco_ids.append(charuco_ids)

    print(f"Number of images used for calibration: {len(all_charuco_corners)}")
    if len(all_charuco_corners) < 5:
        print("Not enough images for calibration")
        quit()
    elif len(all_charuco_corners) != len(all_charuco_ids):
        print("Number of corners and IDs do not match")
        quit()
    else:
        # Calibrate camera
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, image.shape[:2], None, None)

        cv2.destroyAllWindows()
        return camera_matrix, dist_coeffs

def detect_pose(image, camera_matrix, dist_coeffs):
    rvec, tvec = None, None
    # Undistort the image
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

    # Define the aruco dictionary and charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()

    # Detect markers in the undistorted image
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(undistorted_image, dictionary, parameters=params)

    # If at least one marker is detected
    if len(marker_ids) > 0:
        # Interpolate CharUco corners
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, undistorted_image, board)

        # If enough corners are found, estimate the pose
        if charuco_retval:
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None)

            # If pose estimation is successful, draw the axis and save the rvec and tvec
            if retval:
                cv2.drawFrameAxes(undistorted_image, camera_matrix, dist_coeffs, rvec, tvec, length=0.1, thickness=10)
                rvec, tvec = rvec, tvec
    return undistorted_image, rvec, tvec

def check_pose_detection():
    # Iterate through PNG images in the folder
    image_files = [os.path.join(CALIB_DIR_PATH, f) for f in os.listdir(CALIB_DIR_PATH) if f.startswith("intrinsic")]
    image_files.sort()  # Ensure files are in order

    for image_file in image_files:
        # Load an image
        image = cv2.imread(image_file)

        # Detect pose and draw axis
        pose_image, rvec, tvec = detect_pose(image, camera_matrix, dist_coeffs)

        # Show the image
        resized_image = cv2.resize(pose_image, (1000, 800))  # Set the desired size
        cv2.imshow('Pose Image', resized_image)
        cv2.waitKey(0)

def save_intrinsic_parameters(camera_matrix, dist_coeffs, sensor, lens):
    file_path = CALIB_DIR_PATH + sensor + '_' + lens + '_params.json'
    data = {'camera_matrix': camera_matrix.tolist(), 'dist_coeffs': dist_coeffs.tolist()}
    with open(file_path, 'w') as f:
        json.dump(data, f)

create_ChArUco_board()
camera_matrix, dist_coeffs = get_intrinsic_parameters()
check_pose_detection()
save_intrinsics = input("Do you want to save the intrinsic parameters? (y/n): ")
if save_intrinsics.lower() == 'y':
    save_intrinsic_parameters(camera_matrix, dist_coeffs, 'EO-3112C', '25mm-F1.4')


def load_intrinsic_parameters(sensor, lens):
    file_path = CALIB_DIR_PATH + sensor + '_' + lens + '_params.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
        camera_matrix = np.array(data['camera_matrix'])
        dist_coeffs = np.array(data['dist_coeffs'])
    return camera_matrix, dist_coeffs

