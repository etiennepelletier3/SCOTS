# SCOTSpy - Phase measuring deflectometry system
Contact author : etiennepelletier3@gmail.com

## Description
**SCOTSpy** is aimed at providing a fully functionnal phase measuring deflectometry system written in python code.
The system works by displaying vertical and horizontal sinusoidal fringe patterns on a screen and capturing their reflections
onto a surface under test (SUT) using a camera. By analyzing the fringes images, the slopes of the SUT can be recovered, and through
modal integration using zernike polynomials, the surface height map is obtained. It is based on the famous Software Configurable Optical Test System of Su *et al.* (https://doi.org/10.1364/AO.49.004404). If well calibrated, this system should in theory give measurements similar to interferometry and profilometry systems, but at its current stage of development, SCOTSpy still outputs complete shenanigans. It is therefore considered an incomplete project that will be (hopefully) easily fixed with a bit more elbow grease.

![image info](Surface.png)

## List of components
- 1 LCD display (here a laptop screen should suffice)
- 1 Camera (EO-3112C usb camera of Edmund Optics was used, but any camera with the same resolution or higher will do)
- 1 Camera lens (here a 25mm is used)
- 1 flat mirror (used for geometry calibration)
- SUT (can be a parabolic, concave, convex mirror or any reflective surface)
- Optical mounts for mirror, SUT and camera
- 1 printed ArUco marker

## How to use it
### 1. Create new test setup
Create a new Test folder in Calibration data/, Measurement data/ and Results/ folders. Copy the *params.json* file of the previous Results/Test X-1 folder in your Results/Test X folder. When running python files, you will be asked to enter your test ID.

**Side note before moving on** : the terminal gives you a lot of informations about each step taking place, but will sometime be occluded by display windows, so don't forget to take a look at it if you're not sure where you're at.

### 2. Calibrate the camera
Calibrating the camera involves calculating its intrinsic parameters using a chess board. Here a Charuco board is used for enhanced calibration. The camera lens should be adjusted to a certain focus length and aperture size that works for your setup (by having both the SUT and the screen's reflection at a resonnable focus) and those parameters **should not be modified during any subsequent steps**, otherwise the camera will have to be recalibrated. To proceed, place the screen display in front of your camera, and run the *camera_calibration.py* file. You can change parameters like your screen pixel size with *sscreenpx* and marker and grid sizes of the Charuco board. You will be asked if you want to take the poses manually (using an external camera software) or automatically (currently only supported for ueye cameras). Multiple chessboard poses should then be taken (the more the better) and OpenCV will do its magic and provide a camera matrix and a distortion coefficients list which are your camera intrinsics parameters. If taking poses automatically, you can change the number of poses with *nb_poses*. If the calibration went smoothly, you can save the intrinsics parameters in a custom json file with your camera setup name.

### 3. Calibrate the system's geometry
This step is used to measure the position of the components of the system with respect to the camera. At this point, you should place the screen, mirror and camera at there final location, as shown below :

![image info](SCOTS_setup.jpg)

Geometry calibration is done by running the *geometry_calibration.py* file.
 **First**, the pose of the screen display is calculated using the method described in https://doi.org/10.1364/OL.37.000620. Essentially, an Aruco marker is displayed on the screen and a flat mirror is used to reflect its image at the camera. Three different poses of the flat mirror are necessary to extract the pose of the screen with respect to the camera. The user can again choose to go with manual or automatic capture. Look at the terminal to see how to proceed. The screen pose is described by a rotation matrix and a translation vector, the latter being the coordinates of the screen with respect to the camera, which is what we really want. **Second**, replace the reference flat mirror with your SUT, and take another picture of the reflected Aruco marker, which in the code is used for the "zerophase pose point". Basically, the pixel coordinates of the center of the marker in the image corresponds to a reference point present in all the fringe patterns images and is used to obtain the absolute phase from the relative phase later on. **Third**, place a printed Aruco marker carefully on the mirror surface so that the center of the marker is approximately coincident with the SUT's center. You can enter the size of the marker by modifying the *printed_marker_size* variable. Taking a picture will give you the pose of the SUT's origin with respect to the camera. **Note** : this last method is terrible. You have to be carefull not to move the mirror, and for unflat SUT it is almost impossible to make the marker flush with the surface. A possible amelioration would be to create a jig with a preinstalled marker that is placed near the SUT with exact knowledge of its position with respect to the SUT center point. **Finally**, the geometry parameters are saved in *params.json*.

### 4. 

