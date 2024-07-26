from pyueye import ueye
import numpy as np
import cv2

def initialize_camera(exposure_time=33.0, gain=100, framerate=30.0, pixel_clock=5):
    # Initialize the camera
    hCam = ueye.HIDS(0)
    ret = ueye.is_InitCamera(hCam, None)
    if ret != ueye.IS_SUCCESS:
        raise Exception("Failed to initialize camera")

    # Set the color mode to Grayscale
    ueye.is_SetColorMode(hCam, ueye.IS_CM_MONO8)

    # Get sensor info
    sensor_info = ueye.SENSORINFO()
    ueye.is_GetSensorInfo(hCam, sensor_info)
    
    # Get the camera's maximum image size
    width = int(sensor_info.nMaxWidth)
    height = int(sensor_info.nMaxHeight)
    
    # Set the area of interest (AOI) to the entire sensor size
    rect_aoi = ueye.IS_RECT()
    rect_aoi.s32X = ueye.int(0)
    rect_aoi.s32Y = ueye.int(0)
    rect_aoi.s32Width = ueye.int(width)
    rect_aoi.s32Height = ueye.int(height)
    ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))

    # Set the exposure time
    set_exposure(hCam, exposure_time)

    # Set the gain
    set_gain(hCam, gain)

    # Set the framerate
    set_framerate(hCam, framerate)

    # Set the pixel clock (if supported)
    set_pixel_clock(hCam, pixel_clock)

    return hCam, rect_aoi, width, height

def set_exposure(hCam, exposure_time):
    # Set the exposure time
    exposure = ueye.DOUBLE(exposure_time)
    ret = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposure, ueye.sizeof(exposure))
    if ret != ueye.IS_SUCCESS:
        raise Exception("Failed to set exposure time")

def set_gain(hCam, gain):
    # Set the master gain
    ret = ueye.is_SetHardwareGain(hCam, gain, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER)
    if ret != ueye.IS_SUCCESS:
        raise Exception("Failed to set gain")

def set_framerate(hCam, framerate):
    # Set the framerate
    framerate_actual = ueye.DOUBLE(framerate)
    new_framerate = ueye.DOUBLE()
    ret = ueye.is_SetFrameRate(hCam, framerate_actual, new_framerate)
    if ret != ueye.IS_SUCCESS:
        raise Exception("Failed to set framerate")
    print(f"Actual framerate set: {new_framerate.value} fps")

def set_pixel_clock(hCam, pixel_clock):
    # Set the pixel clock
    clock = ueye.UINT(pixel_clock)
    ret = ueye.is_PixelClock(hCam, ueye.IS_PIXELCLOCK_CMD_SET, clock, ueye.sizeof(clock))
    if ret != ueye.IS_SUCCESS:
        raise Exception("Failed to set pixel clock")
    print(f"Pixel clock set to: {pixel_clock} MHz")

def capture_frame(hCam, width, height):
    # Allocate memory for the image
    bitspixel = 8  # for grayscale mode: 8 bits per pixel
    mem_ptr = ueye.c_mem_p()
    mem_id = ueye.int()
    ret = ueye.is_AllocImageMem(hCam, width, height, bitspixel, mem_ptr, mem_id)
    if ret != ueye.IS_SUCCESS:
        raise Exception("Failed to allocate image memory")
    ret = ueye.is_SetImageMem(hCam, mem_ptr, mem_id)
    if ret != ueye.IS_SUCCESS:
        raise Exception("Failed to set image memory")
    ret = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)
    if ret != ueye.IS_SUCCESS:
        raise Exception("Failed to set display mode")

    # Capture an image
    ret = ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)
    if ret != ueye.IS_SUCCESS:
        raise Exception("Failed to freeze video")

    # Extract image data
    array = ueye.get_data(mem_ptr, width, height, bitspixel, width, copy=True)
    frame = np.reshape(array, (height, width))

    # Free the image memory
    ueye.is_FreeImageMem(hCam, mem_ptr, mem_id)

    return frame

def main():
    exposure_time = 100.0  # Set your desired exposure time here (in milliseconds)
    gain = 50              # Set your desired gain here
    framerate = 30.0       # Set your desired framerate here (in fps)
    hCam, rect_aoi, width, height = initialize_camera(exposure_time, gain, framerate)

    cv2.namedWindow("Live Video", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frame = capture_frame(hCam, width, height)
            cv2.imshow("Live Video", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release the camera and close OpenCV windows
        ueye.is_ExitCamera(hCam)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


# import cv2
# import numpy as np
# import json

# img1 = cv2.imread("Calibration data/Test 3/pose1.png")
# img2 = cv2.imread("Calibration data/Test 2/pose_02.png")
# print(img1.shape)
# print(img2.shape)

# ARUCO_DICT = cv2.aruco.DICT_4X4_50  # dictionary ID
# def poseEstimation(img, marker_size, camera_matrix, dist_coeffs):
#     undist_img = cv2.undistort(img, camera_matrix, dist_coeffs)
#     dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
#     params = cv2.aruco.DetectorParameters()
#     # params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

#     corners, ids, rejected = cv2.aruco.detectMarkers(undist_img, dictionary, parameters=params)

#     marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
#                               [marker_size / 2, marker_size / 2, 0],
#                               [marker_size / 2, -marker_size / 2, 0],
#                               [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    
#     if len(corners) > 0:
#         print(f"Corner: {corners}")
#         for corner in corners:
#             _, rvec, tvec = cv2.solvePnP(marker_points, corner, camera_matrix, dist_coeffs, None, None, False, cv2.SOLVEPNP_IPPE_SQUARE)
#             # Calculate the distance between the corner points in the image (in pixels)
#             width = np.linalg.norm(corner[0][0] - corner[0][1])
#             height = np.linalg.norm(corner[0][1] - corner[0][2])
#             avg_pixel_length = (width + height) / 2

#             origin_point_3d = np.array([[0, 0, 0]], dtype='float32')  # The origin in the world coordinate system
#             origin_point_2d, _ = cv2.projectPoints(origin_point_3d, rvec, tvec, camera_matrix, dist_coeffs) # Project the origin point to the image frame
#             center = (origin_point_2d[0][0][0], origin_point_2d[0][0][1]) # Convert to integer tuple4
#             center_int = (int(center[0]), int(center[1]))
#             cv2.circle(undist_img, center_int, 1, (0, 0, 255), -1)
#             cv2.aruco.drawDetectedMarkers(undist_img, corners, borderColor=(0, 255, 0))
#             cv2.drawFrameAxes(undist_img, camera_matrix, dist_coeffs, rvec, tvec, length=marker_size/2, thickness=2)

#     return undist_img, center, rvec, tvec, avg_pixel_length

# intrinsics_path = 'Calibration data/EO-3112C_25mm-F1.4_params.json'

# # Load calibration parameters from JSON file
# with open(intrinsics_path, 'r') as file: # Read the JSON file
#     json_data = json.load(file)
# camera_matrix = np.array(json_data['camera_matrix']) # Load the camera matrix
# dist_coeffs = np.array(json_data['dist_coeffs']) # Load the distortion coefficients

# mrk_size = 0.223*350/1000
# img, center, rvec, tvec, avg_pixel_length = poseEstimation(img1, mrk_size, camera_matrix, dist_coeffs)

# print(f"Center: {center}")
# print(f"rvec: {rvec}")
# print(f"tvec: {tvec}")
# print(f"avg_pixel_length: {avg_pixel_length}")
# cv2.imshow("Image", img)
# cv2.waitKey(0)

# img, center, rvec, tvec, avg_pixel_length = poseEstimation(img2, mrk_size, camera_matrix, dist_coeffs)

# print(f"Center: {center}")
# print(f"rvec: {rvec}")
# print(f"tvec: {tvec}")
# print(f"avg_pixel_length: {avg_pixel_length}")
# cv2.imshow("Image", img)
# cv2.waitKey(0)