# SCOTSpy - Phase measuring deflectometry system

### Description
**SCOTSpy** is aimed at providing a fully functionnal phase measuring deflectometry system written in python code.
The system works by displaying vertical and horizontal sinusoidal fringe patterns on a screen and capturing their reflections
onto a surface under test (SUT) using a camera. By analysing the fringes images, the slopes of the SUT can be recovered, and through
modal integration using zernike polynomials, the surface height map is obtained. If well calibrated, this system should in theory give measurements similar to interferometry and profilometry systems, but at its current stage of development, SCOTSpy still outputs complete shenanigans. It is therefore considered an incomplete project that can be (hopefully) easily fixed with a bit more elbow grease.

### List of components
- 1 LCD display (here a laptop screen should suffice)
- 1 Camera (EO-3112C usb camera of Edmund Optics was used, but any camera with the same resolution or higher will do)
- 1 flat mirror (used for geometry calibration)
- SUT (can be a parabolic, concave, convex mirror or any reflective surface)
- Optical mounts for mirror, SUT and camera

## How to use it

