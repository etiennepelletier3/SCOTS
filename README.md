# SCOTSpy - Phase measuring deflectometry system

## Description
**SCOTSpy** is aimed at providing a fully functionnal phase measuring deflectometry system written in python code.
The system works by displaying vertical and horizontal sinusoidal fringe patterns on a screen and capturing their reflections
onto a surface under test (SUT) using a camera. By analysing the fringes images, the slopes of the SUT can be recovered, and through
modal integration using zernike polynomials, the surface height map is obtained. If well calibrated, this system should in theory give measurements similar to interferometry and profilometry systems, but at its current stage of development, SCOTSpy still outputs complete shenanigans. It is therefore considered an incomplete project that will be (hopefully) easily fixed with a bit more elbow grease.

![image info](Surface.png)

## List of components
- 1 LCD display (here a laptop screen should suffice)
- 1 Camera (EO-3112C usb camera of Edmund Optics was used, but any camera with the same resolution or higher will do)
- 1 Camera lens (Here a 25mm is used)
- 1 flat mirror (used for geometry calibration)
- SUT (can be a parabolic, concave, convex mirror or any reflective surface)
- Optical mounts for mirror, SUT and camera

## How to use it
### 1. Create new test setup
Create a new Test folder in Calibration data/, Measurement data/ and Results/ folders. Copy the *params.json* file of the previous Results/Test X-1 folder in your Results/Test X folder. When running python files, you will be asked to enter your test ID.

### 2. Calibrate the camera
Calibrating the camera involves calculating its intrinsic parameters using a chessboard. Here a Charuco board is used for enhanced calibration. The camera lens should be adjusted to a certain focus length and aperture size that works for your setup (by having both the SUT and the screen's reflection at a resonnable focus) and those parameters **should not be modified during any subsequent steps**, otherwise the camera will have to be recalibrated. To proceed, place the screen display in front of your camera, and run the *camera_calibration.py* file. You can change parameters like your screen pixel size with *sscreenpx* and marker and grid sizes of the Charuco board. You will be asked if you want to take the poses manually (using an external camera software) or automatically (currently only supported for ueye cameras). Multiple chessboard poses should then be taken (the more the better) and OpenCV will do its magic and provide a camera matrix and a distortion coefficients list which are your camera intrinsics parameters. If taking poses automatically, you can change the number of poses with *nb_poses*.
