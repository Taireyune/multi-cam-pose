import numpy as np
import cv2, os, h5py
from global_objects import GlobalConfig as G
from cv_reconstruct.cam_tools3 import read_calibration, rectify_checker_grid
from cv_reconstruct.cam_tools3 import corner_orientation

def undistort(points, mtx, dist, point_count = 1):    
    if point_count == 1:
        points = points[None, None, :].astype(np.float32)
    else:
        points = points[:, None, :].astype(np.float32)
    # undistortPoints uses shape (N, 1, 2) where N is number of points and 2 contain the x and y    
    points = cv2.undistortPoints(points, mtx, dist)
    return points.squeeze()

class dual_cam_triangulation:
    def __init__(self, calibration_dir, point_count = 1):
        
        parameters = read_calibration(calibration_dir)
        self.mtx_1 = parameters[0]
        self.mtx_2 = parameters[1]
        self.dist_1 = parameters[2]
        self.dist_2 = parameters[3]
        self.P_1 = parameters[4]
        self.P_2 = parameters[5]
        self.point_count = point_count
        
        print("Calibration projection error:", parameters[6])
  
        self.rearrange = np.array([0,2,1], dtype = int)
        
        
    def triangulate(self, pt_1, pt_2):
        pt_1 = undistort(pt_1, self.mtx_1, self.dist_1, point_count = self.point_count)        
        pt_2 = undistort(pt_2, self.mtx_2, self.dist_2, point_count = self.point_count)
        
        t_points = cv2.triangulatePoints(self.P_1, self.P_2, pt_1.T, pt_2.T)

        points = t_points[:3,]/t_points[-1,:] 
        
        return points.T
#        points = points.T
#        return points[:, self.rearrange]         # Y and Z vectors are swapped
    
class dual_cam_calibration:
    def __init__(self, grid_shape, grid_distance, mtx_dir_1, mtx_dir_2, key_frames, image_dir):
        #initialize grid and values
        '''
        add mtx_dir stuff
        
        '''
        self.grid_shape = grid_shape
        self.key_frames = key_frames
        self.image_dir = image_dir #image_dir must contain 2 {} to identy camera ID and key_frame
        
        #create objective point grid
        self.objp = rectify_checker_grid(self.grid_shape, distance = grid_distance)
        
        # criteria
        self.corner_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.stereo_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6)

        #self.calibration_flags = (cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL)       
        #self.calibration_flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_FOCAL_LENGTH + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL)
        self.calibration_flags = (cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL)
        
        # Arrays to store object points and image points from all the images.
        self.obj_points = [] # 3d point in real world space
        self.img_points_1 = [] # 2d points in image plane.
        self.img_points_2 = []
        self.good_frames = []
        
        # save file name
        file_name = "parameters.h5"
        self.file_name = os.path.join(G.folder_dir, file_name)
        
        # read video directory
        #self.video_dir_1 = "/media/taireyune/RAIDext4/video_folder/reconstruct1.avi"
        #self.video_dir_2 = "/media/taireyune/RAIDext4/video_folder/reconstruct2.avi"
        self.video_dir_1 = G.video_dir.format(0, 1)
        self.video_dir_2 = G.video_dir.format(1, 1)
          
        # initialize successful read count
        self.image_hit = 0 
        
        self.read_video()
        self.load_parameters(mtx_dir_1, mtx_dir_2)
        self.compute_matrix()
    
    def read_video(self):
        
        cap_1 = cv2.VideoCapture(self.video_dir_1)
        cap_2 = cv2.VideoCapture(self.video_dir_2)   
        self.count = 1
        while True:
            cap_1_running, image_1 = cap_1.read()
            cap_2_running, image_2 = cap_2.read()
        
            if cv2.waitKey(1) == ord('q') or self.image_hit == 100 or cap_1_running == False or cap_2_running == False:
                print("stop reading video")        
                break   
            
            if self.count in self.key_frames:
                self.get_points(image_1, image_2)
            
            self.count += 1         
                
    def get_points(self, raw_image_1, raw_image_2):
        gray_1 = cv2.cvtColor(raw_image_1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(raw_image_2, cv2.COLOR_BGR2GRAY)
        ret_1, corner_1 = cv2.findChessboardCorners(gray_1, self.grid_shape, None)
        ret_2, corner_2 = cv2.findChessboardCorners(gray_2, self.grid_shape, None)
        
        if ret_1 and ret_2:
            self.obj_points.append(self.objp)
            corner_1 = cv2.cornerSubPix(gray_1, corner_1, (11,11), (-1,-1), self.corner_criteria)
            corner_2 = cv2.cornerSubPix(gray_2, corner_2, (11,11), (-1,-1), self.corner_criteria)
            
            corner_1 = corner_orientation(corner_1, self.grid_shape)
            corner_2 = corner_orientation(corner_2, self.grid_shape)
            
            gray_1 = cv2.drawChessboardCorners(raw_image_1, self.grid_shape, corner_1, True)
            gray_2 = cv2.drawChessboardCorners(raw_image_2, self.grid_shape, corner_2, True)
            
            cv2.imwrite(self.image_dir.format(self.count, 0), gray_1)
            cv2.imwrite(self.image_dir.format(self.count, 1), gray_2)  
            
            self.good_frames.append(self.count)
            self.img_points_1.append(corner_1)
            self.img_points_2.append(corner_2)
            
            self.image_hit += 1
            print("count: {}, imageHit: {}".format(self.count, self.image_hit)) 
            
        elif ret_1:
            corner_1 = cv2.cornerSubPix(gray_1, corner_1, (11,11), (-1,-1), self.corner_criteria)            
            corner_1 = corner_orientation(corner_1, self.grid_shape)            
            gray_1 = cv2.drawChessboardCorners(raw_image_1, self.grid_shape, corner_1, True) 
            
            cv2.imwrite(self.image_dir.format(self.count, 0), gray_1)
            cv2.imwrite(self.image_dir.format(self.count, 1), raw_image_2)
            print("count: {}, detection failed for cam_2".format(self.count))    
            
        elif ret_2:
            corner_2 = cv2.cornerSubPix(gray_2, corner_2, (11,11), (-1,-1), self.corner_criteria)
            corner_2 = corner_orientation(corner_2, self.grid_shape)
            gray_2 = cv2.drawChessboardCorners(raw_image_2, self.grid_shape, corner_2, True)
            
            cv2.imwrite(self.image_dir.format(self.count, 0), raw_image_1)
            cv2.imwrite(self.image_dir.format(self.count, 1), gray_2)  
            print("count: {}, detection failed for cam_1".format(self.count))    
            
        else:
            cv2.imwrite(self.image_dir.format(self.count, 0), raw_image_1)
            cv2.imwrite(self.image_dir.format(self.count, 1), raw_image_2)
            print("count: {}, detection failed".format(self.count))            
            
    def compute_matrix(self):
        print("computing dual camera parameters...")
        # build P_1
        R = np.identity(3, dtype=np.float32)
        T = np.zeros((3,1), dtype=np.float32)        
        P_1 = np.concatenate((R, T), axis = 1)
        
        ret, mtx_1, dist_1, mtx_2, dist_2, R, T, E, F = cv2.stereoCalibrate( 
                self.obj_points, self.img_points_1, self.img_points_2, 
                self.mtx_1, self.dist_1, self.mtx_2, self.dist_2, 
                (G.img_width, G.img_height), 
                flags = self.calibration_flags,
                criteria = self.stereo_criteria)  
        
        P_2 = np.concatenate((R, T), axis = 1)  
        print("matrix_1:", "\n",  
              mtx_1, "\n", 
              "mtx_2:", "\n", 
              mtx_2, "\n",                            
              "dist_1:", "\n", 
              dist_1, "\n", 
              "dist_2:", "\n", 
              dist_2,"\n", 
              "P_1:", "\n", 
              P_1,"\n", 
              "P_2:", "\n", 
              P_2)
        if ret:
            print("RMS reprojection error:", ret)
            print("good frames:", self.good_frames)
            with h5py.File(self.file_name, 'w') as write_file:
                # camera matrix and distortion matrix from stereo calibration
                write_file.create_dataset('mtx_1', data = mtx_1)
                write_file.create_dataset('mtx_2', data = mtx_2)   
                write_file.create_dataset('dist_1', data = dist_1)
                write_file.create_dataset('dist_2', data = dist_2) 
                
                # R and P from stereorectify                                 
                write_file.create_dataset('P_1', data = P_1)
                write_file.create_dataset('P_2', data = P_2)
                
                # error and key_frames
                write_file.create_dataset('error', data = [ret])
                write_file.create_dataset('key_frames', data = self.good_frames)    
                    
    def load_parameters(self, mtx_dir_1, mtx_dir_2):
        with h5py.File(mtx_dir_1, 'r') as open_file:    
            self.mtx_1 = open_file['mtx'][:]
            self.dist_1 = open_file['dist'][:] 
            
        with h5py.File(mtx_dir_2, 'r') as open_file:    
            self.mtx_2 = open_file['mtx'][:]
            self.dist_2 = open_file['dist'][:]         
        print("single cam parameters loaded")






























        