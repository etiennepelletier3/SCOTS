from pyueye import ueye
import numpy as np
import cv2

def initialize_camera(exposure_time, gain, framerate, pixel_clock):
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

    # Set the pixel clock
    set_pixel_clock(hCam, pixel_clock)

    return hCam, rect_aoi, width, height

def set_exposure(hCam, exposure_time):
    # Set the exposure time
    exposure = ueye.DOUBLE(exposure_time)
    ret = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposure, ueye.sizeof(exposure))
    if ret != ueye.IS_SUCCESS:
        raise Exception("Failed to set exposure time")
    print(f"Exposure time set: {exposure_time} ms")

def set_gain(hCam, gain):
    # Set the master gain
    ret = ueye.is_SetHardwareGain(hCam, gain, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER)
    if ret != ueye.IS_SUCCESS:
        raise Exception("Failed to set gain")
    print(f"Gain set: {gain}")

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


def adjust_camera_parameters():
    exposure_time = 275.0  # Set your desired exposure time here (in milliseconds)
    gain = 100           # Set your desired gain here
    framerate = 1.0       # Set your desired framerate here (in fps)
    pixel_clock = 17      # Set your desired pixel clock here (in MHz)
    hCam, rect_aoi, width, height = initialize_camera(exposure_time, gain, framerate, pixel_clock)

    try:
        while True:
            frame = capture_frame(hCam, width, height)
            # frame = adjust_contrast(frame, alpha=contrast_alpha, beta=contrast_beta)
            frame = cv2.resize(frame, (640, 480))
            cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
            cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)
            
            cv2.imshow("Camera", frame)

            # Capture keyboard events
            key = cv2.waitKey(1) & 0xFF

            # Update gain, exposure, and frame rate based on keyboard input
            if key == ord('g'):
                gain += 5
                if gain > 100:
                    gain = 100
                set_gain(hCam, gain)
                print(f"Gain set: {gain}")
            elif key == ord('b'):
                gain -= 5
                if gain < 0:
                    gain = 0
                set_gain(hCam, gain)
                print(f"Gain set: {gain}")
            elif key == ord('e'):
                exposure_time += 10.0
                if exposure_time > 600.0:
                    exposure_time = 600.0
                set_exposure(hCam, exposure_time)
            elif key == ord('d'):
                exposure_time -= 25.0
                if exposure_time < 0.2:
                    exposure_time = 0.2
                set_exposure(hCam, exposure_time)
            elif key == ord('f'):
                framerate += 1.0
                if framerate > 10.0:
                    framerate = 10.0
                set_framerate(hCam, framerate)
                print(f"Framerate set: {framerate} fps")
            elif key == ord('v'):
                framerate -= 1.0
                if framerate < 1.0:
                    framerate = 1.0
                set_framerate(hCam, framerate)
                print(f"Framerate set: {framerate} fps")
            elif key == ord('o'):
                pixel_clock += 1
                if pixel_clock > 43:
                    pixel_clock = 43
                set_pixel_clock(hCam, pixel_clock)
            elif key == ord('l'):
                pixel_clock -= 1
                if pixel_clock < 5:
                    pixel_clock = 5
                set_pixel_clock(hCam, pixel_clock)
            elif key == ord('q'):
                break

    finally:
        # Release the camera and close OpenCV windows
        ueye.is_ExitCamera(hCam)
        cv2.destroyAllWindows()
        print("Gain set: ", gain)
        print("Exposure time set: ", exposure_time)
        print("Framerate set: ", framerate)
        return gain, exposure_time, framerate, pixel_clock
    

if __name__ == "__main__":
    gain, exposure_time, framerate, pixel_clock = adjust_camera_parameters()
    hCam, rect_aoi, width, height = initialize_camera(exposure_time, gain, framerate, pixel_clock)
    

