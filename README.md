# multi_cam_pose

**Multi Cam Pose** uses
[2D real-time pose estimation]()
with multiple cameras (currently only made to work with my own setup) to 
triangulate 3D human key point coordinates in real time.

It is written with python-multiprocessing, numpy, cv2, and the pose estimation by 
[ildoonet]() 
in tensorflow. This repository contains some sample codes. 

## Work flow
To begin, the know-how and methods for the camera calibration and 3D triangulation 
processes came from [OpenCV](https://docs.opencv.org/3.4.3/). 
So glad they have everything I needed in C++ and python open-sourced.

1. [**Single camera calibration**]()
2. [**Multi camera calibration**]()
3. [**Coordinate system definition**]()
4. [**Results**]()
5. [**Up next**]()

## Single camera calibration
### Select frames from video
<img 
src="   .png" 
alt="single camera GUI">

### Check detection quality and reprojection error
The images are fed to the 
[single_cam.py method]()
to obtain the detected corners and reprojection errors for trouble shooting and
calibration quality control. If everything checks out, the output
camera parameters are used for the next step.

<img 
src="   .gif" 
alt="single camera detection">

## Multi camera calibration
[dual_cam.py methods]()
is used to find spatial relationships between the two cameras. 
From an operating point of view, the calibration process is the same as the
single camera calibration process, but requires fewer images to get the job
done. 

<img 
src="   .gif" 
alt="dual camera detection">

## Coordinate system definition
Camera parameters from the previous two methods along with two video feed
are used to triangulate object in 3D space. I used the position of my left 
arm in the beginning of each stream to define the center of the 
[viewing window]()
and the upright direction of the coordinate system.

<img 
src="   .png" 
alt="set system">

## Here are the results
<img 
src="   .gif" 
alt="set system">

## Up next
### Robust and automated calibration
### Better viewing window and depth perception
### Inference model