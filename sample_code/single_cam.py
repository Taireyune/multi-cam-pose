import numpy as np
import cv2, os, h5py
from global_objects import GlobalConfig as G
from cv_reconstruct.cam_tools3 import rectify_checker_grid, corner_orientation

#get points from checkerboard corners
def single_cam(grid_shape, grid_distance, file_name, video_dir, image_dir, key_frames):
    # directories  
    file_name = os.path.join(G.folder_dir, file_name)
    
    # corner t criteria
    corner_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
    
    # calibration flags 
    calibration_flags = (cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL)

    #create objective grid points
    objp = rectify_checker_grid(grid_shape, distance = grid_distance)
    
    # Arrays to store object points and image points from all the images.
    objPoints = [] # 3d point in real world space
    imgPoints = [] # 2d points in image plane.
    good_frames = []
    # counters
    imageHit = 0     # successful corner reading
    count = 1        # number of frames read
    
    cap = cv2.VideoCapture(video_dir)
    
    while True:
        playing_video, rawImage = cap.read()
        
        if cv2.waitKey(1) == ord('q') or imageHit == 100 or playing_video == False:
            print("stop video read")        
            break    
        
        if count in key_frames:
            grayImage = cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret1, cornerCoordinate = cv2.findChessboardCorners(grayImage, grid_shape, None)    
            
            # If found, add object points, image points (after refining them)
            if ret1 == True: 
                cornerCoordinate = cv2.cornerSubPix(grayImage, cornerCoordinate, (11,11), (-1,-1), corner_criteria)
                #print("before:", cornerCoordinate)
                cornerCoordinate = corner_orientation(cornerCoordinate, grid_shape)
                #print("after:", cornerCoordinate)
                objPoints.append(objp)
                imgPoints.append(cornerCoordinate)
                imageHit += 1
                
                # image_dir must contain {} for labeling count
                image = cv2.drawChessboardCorners(rawImage, grid_shape, cornerCoordinate, True)
                cv2.imwrite(image_dir.format(count), image)
                good_frames.append(count)
                print("count: {}, imageHit: {}".format(count, imageHit))  
            else:
                cv2.imwrite(image_dir.format(count), rawImage)
                print("count: {}, detection failed".format(count))

        count += 1
        
    try:
        print("compute calibrations...")
        ret2, mtx, dist, _, _ = cv2.calibrateCamera(objPoints, imgPoints, 
                                                    grayImage.shape[::-1], 
                                                    None, None, 
                                                    flags = calibration_flags)
    except Exception as e:
        print(e)
        print("calibration failed")
        return         

    if ret2:
        print("Calibration_1 RMS reprojection error:", ret2)
        print('Good frames:', good_frames)
        print(mtx, '\n', dist)
        with h5py.File(file_name, 'w') as write_file:
            write_file.create_dataset('mtx', data = mtx)
            write_file.create_dataset('dist', data = dist)
            write_file.create_dataset('error', data = ret2)
            write_file.create_dataset('key_frames', data = np.array(good_frames))

    else:
        print("calibration failed")
        return        
    

#%% get undistorted image for debug
# undistort
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.png',dst)

            