# multi-cam-pose

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

1. [**Single camera calibration**](https://github.com/Taireyune/multi-cam-pose#single-camera-calibration)
2. [**Multi camera calibration**](https://github.com/Taireyune/multi-cam-pose#multi-camera-calibration)
3. [**Coordinate system definition**](https://github.com/Taireyune/multi-cam-pose#coordinate-system-definition)
4. [**Results**](https://github.com/Taireyune/multi-cam-pose#here-are-the-results)
5. [**Up next**](https://github.com/Taireyune/multi-cam-pose#up-next)

## Single camera calibration
### Select frames from video
<img 
src="https://github.com/Taireyune/multi-cam-pose/blob/master/images/cam_GUI.png" 
width="810" height="500" alt="single camera GUI">

### Check detection quality and reprojection error
The images are fed to the 
[single_cam.py](https://github.com/Taireyune/multi-cam-pose/blob/master/sample_code/single_cam.py)
to obtain the detected corners and reprojection errors for trouble shooting and
calibration quality control. If everything checks out, the output
camera parameters are used for the next step.

<img 
src="https://github.com/Taireyune/multi-cam-pose/blob/master/images/single_cam.gif" 
width="810" height="689" alt="single camera detection">

## Multi camera calibration
[dual_cam.py](https://github.com/Taireyune/multi-cam-pose/blob/master/sample_code/dual_cam.py)
is used to find spatial relationships between the two cameras. 
From an operating point of view, the calibration process is the same as the
single camera calibration process, but requires fewer images to get the job
done. 

<img 
src="https://github.com/Taireyune/multi-cam-pose/blob/master/images/dual_cam.gif" 
width="810" height="1379" alt="dual camera detection">

## Coordinate system definition
Camera parameters from the previous two methods along with two video feed
are used to triangulate object in 3D space. I used the position of my left 
arm in the beginning of each stream to define the center of the 
[viewing window](https://github.com/Taireyune/multi-cam-pose/blob/master/sample_code/view_window.py)
and the upright direction of the coordinate system.

<img 
src="https://github.com/Taireyune/multi-cam-pose/blob/master/images/set_system.png" 
width="810" height="2068" alt="set system">

## Here are the results
### Side view
Here is the 3D view from the right side of the cameras.

<img 
src="https://github.com/Taireyune/multi-cam-pose/blob/master/images/side_view.gif" 
width="500" height="1277" alt="side view">

### Top view
I pointed at where the camera projection would be.

<img 
src="https://github.com/Taireyune/multi-cam-pose/blob/master/images/top_view.gif" 
width="500" height="1277" alt="top view">

### Real time
For real time inference, I had to run a lot of the methods in parallel and
down sample the video frames: 
[link to master_parallel.py](https://github.com/Taireyune/multi-cam-pose/blob/master/sample_code/master_parallel.py).

Although I say real time, there is some latency between the captured frames and
the displayed 3D inference. The batch speed of inference are the main source of 
the latency.

The method can also handle writing video to disk while real-time inference 
is running. It saves the video at a much higher frame rate so that a better
quality inference can be done post-capture 
(my system write the videos on RAID disks, so disk-io is not a bottle-neck).

Really gotta be more fluent in C++ if I want to get more performance out...

<img 
src="https://github.com/Taireyune/multi-cam-pose/blob/master/images/live_display.gif" 
width="810" height="542" alt="live display">

## Up next
### Robust and automated calibration
The current setup requires me to check each calibration image to find bad
corner detections. I am working towards a more streamlined calibration process.

### Better viewing window and depth perception
I did not use a standard GUI such as the matplotlib 3D graphs for the view
window because it does not run properly in a single subprocess. The current cv2 native 
view window is lightweight and can run in parallel with the other processes 
but lack the intuitive controls. 

Something we usually take for granted in a 3D viewers is the depth perception 
of points and lines. Currently, all the points and lines are the same width
regardless of distance from the reprojection plane. The reprojection of me 
currently doesn't look very 3D. 

### Inference model
There are still a lot of improvements we can make with the pose estimation 
engine. I am building my own to try out some of the ideas I haven't seen 
in the papers. We'll see how that goes.
