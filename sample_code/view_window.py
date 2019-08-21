import cv2
import numpy as np
from global_objects import GlobalConfig as G

class view_window:
    def __init__(self, reference_list = [6, 7, 5], sample_frames = 10):        
        self.window_name = "3D view"
        
        # projection parameters
        self.mtx = G.view_mtx
        
        # initialize transformation vectors
#        self.rotation = np.array([[0, 0, -np.pi/2]], dtype=np.float32)
        self.rotation = np.zeros((1,3), dtype=np.float32)
        self.translation = np.zeros((1,3), dtype=np.float32)
        
        # sampling points 
        self.sample_frames = sample_frames
        self.reference_list = reference_list
        
        '''
        Default 6, 7, 5 are the index for LElbow, LWrist, LShoulder. 
        The first value will define the origin. 
        The direction of the first and third value define the X direction.
        The direction of the plane formed by the three points is the Y direction.
        sample_pts will collect the points from each index within the given 
        sample_frames and take an average.
        The new coordinate system will be based on these three averages.
        The function will only collect data from the first person that shows 
        the first (smallest index) reference point.
        '''
        
        self.reference = np.zeros((3,3), dtype = np.float32)
        self.reference_count = np.zeros((3, 1), dtype = int)
        self.reference_person = -1
        self.frame_count = 0
        
        ''' 
        initialize coordinate system
        Y and Z vectors are swapped and both become negative to ensure a view
        in the upright direction.   
        '''
        self.axes = np.array([[0, 0, 0],  
                                                                                       
                              [0.25, 0, 0],
                              [0.125, 0.02, 0],
                              [0.125, -0.02, 0],
                              [0.25, 0.02, 0],
                              [0.25, -0.02, 0],  
                              
                              [0, 0, -0.25],
                              [0.02, 0, -0.125],
                              [-0.02, 0, -0.125],
                              [0.02, 0, -0.25],
                              [-0.02, 0, -0.25],
                              
                              [0, -0.25, 0],
                              [0, -0.125, 0.02],
                              [0, -0.125, -0.02],
                              [0, -0.25, 0.02],
                              [0, -0.25, -0.02],], 
                              dtype = np.float32) 
        
#        self.axes = np.array([[0, 0, 0],  
#                                                                                       
#                              [0.25, 0, 0],
#                              [0.125, 0.02, 0],
#                              [0.125, -0.02, 0],
#                              [0.25, 0.02, 0],
#                              [0.25, -0.02, 0],  
#                              
#                              [0, 0.25, 0],
#                              [0, 0.125, 0.02],
#                              [0, 0.125, -0.02],
#                              [0, 0.25, 0.02],
#                              [0, 0.25, -0.02],                                
#                              
#                              [0, 0, 0.25],
#                              [0.02, 0, 0.125],
#                              [-0.02, 0, 0.125],
#                              [0.02, 0, 0.25],
#                              [-0.02, 0, 0.25]], 
#                              dtype = np.float32) 
        
        self.axes_pairs = [(0, 1), (2, 3), (4, 5), 
                           (0, 6), (7, 8), (9, 10),
                           (0, 11), (12, 13), (14, 15)]
                           
              
        # initialize view controls
        self.r_increment = np.pi/24
        self.t_increment = 0.1
        
        self.left_threshold = G.img_width // 3  # set control threshold
        self.right_threshold = G.img_width * 2 // 3
        self.rt_threshold = G.img_height // 2
          
        # initialize cv2 GUI 
        cv2.namedWindow(self.window_name , cv2.WINDOW_AUTOSIZE + cv2.WINDOW_GUI_NORMAL )
        
        cv2.setMouseCallback(self.window_name, self.call_back)
        self.image = np.zeros(G.Ishape, dtype = np.uint8)  
#        self.draw_view(np.zeros((G.max_persons * G.part_count, 3), dtype = np.float32),
#                       np.zeros((G.max_persons * G.part_count, 1), dtype = bool))
        cv2.imshow(self.window_name, self.image)     
        # add coordinates
        
    def run(self, points_3D, mask):
        
        if self.frame_count == -1:
            self.draw_view(points_3D, mask)
            
        elif self.frame_count < self.sample_frames:
            self.sample_pts(points_3D, mask)
            
        else:           
            self.orient_view()

            print(self.translation, '\n', self.reference_count)
            # ends point sampling
            self.frame_count = -1
        
    def sample_pts(self, points_3D, mask):
        points_3D = points_3D.reshape(G.max_persons, G.part_count, 3) 
        mask = mask.reshape(G.max_persons, G.part_count) 
        for per in range(G.max_persons):
            if self.reference_person not in (per, -1):
                continue
            for p in range(G.part_count):
                if not mask[per, p]:
                    continue
                if p not in self.reference_list:
                    continue
                i = self.reference_list.index(p)
                self.reference[i, 0] += points_3D[per, p, 0]
                self.reference[i, 1] += points_3D[per, p, 1]
                self.reference[i, 2] += points_3D[per, p, 2]                
                self.reference_count[i,0] += 1
                self.reference_person = per
        self.frame_count += 1
        cv2.imshow(self.window_name, self.image)
        
    def orient_view(self):
        # upright coordinate system
        # adjust axes by changing the index 1, 6, 11
        up_sys = np.vstack((self.axes[0], self.axes[11], self.axes[1]))
        up_sys = prep_values(up_sys)
        
        # camera coordinate system
        self.reference /= self.reference_count
        cam_sys = prep_values(self.reference)  

        # Get homogeneous affine transformation matrix that takes points from cam to an upright coordinate system
        self.R, self.t = homogenous_transformation(cam_sys, up_sys)
        
        # move camera
        self.translation -= self.reference[0]

    def draw_view(self, points_3D, mask):
        image = self.image.copy()
        image = self.draw_pose(image, points_3D, mask)
        image = self.draw_axes(image)
        
        cv2.imshow(self.window_name, np.flip(image, axis = 0))
        
    def draw_axes(self, image):
        axes, _ = cv2.projectPoints(self.axes,
                                    self.rotation,
                                    self.translation,
                                    self.mtx, 0)
        axes = axes.reshape(16,2)
        for idx, pair in enumerate(self.axes_pairs):
            cv2.line(image, 
                     (axes[pair[0], 0], axes[pair[0], 1]), 
                     (axes[pair[1], 0], axes[pair[1], 1]),
                     G.draw_colors[(idx // 3) * 6], 1)
        return image
    
    def draw_pose(self, image, points_3D, mask):
        # from cam coordinates to upright coordinates
        points_3D = np.dot(points_3D, self.R) + self.t
        points_2D, _ = cv2.projectPoints(points_3D, 
                                         self.rotation, 
                                         self.translation, 
                                         self.mtx, 0)
        
        points_2D = points_2D.reshape(G.kp_shape) 
        mask = mask.reshape(G.max_persons, G.part_count)     
 
        # loops to draw the points
        for per in range(G.max_persons):
            
            # Draw points. Image is modified
            for p in range(G.part_count):
                if not mask[per, p]:
                    continue
                #if points_2D[per, p, 0]
                cv2.circle(image, 
                           (points_2D[per, p, 0], points_2D[per, p, 1]),
                           3, G.draw_colors[p], thickness = 2, 
                           lineType = 8, shift = 0)
                
            # Draw lines. Image is modified
            for pair_order, p in enumerate(G.part_pairs):
                if not mask[per, p[0]] or not mask[per, p[1]]:
                    continue
                cv2.line(image, 
                         (points_2D[per, p[0], 0], points_2D[per, p[0], 1]), 
                         (points_2D[per, p[1], 0], points_2D[per, p[1], 1]),
                         G.draw_colors[pair_order], 3)
        return image
        
    def call_back(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.up_dial(x, y)
            
        if event == cv2.EVENT_RBUTTONDBLCLK :
            self.down_dial(x, y)
            
        if event == cv2.EVENT_MBUTTONDBLCLK:
            self.reset_dial()
    
    def up_dial(self, x, y):
        if y <= self.rt_threshold:
            if x <= self.left_threshold:
                self.rotation[0, 0] += self.r_increment
            elif self.left_threshold < x < self.right_threshold:
                self.rotation[0, 1] += self.r_increment
            else:
                self.rotation[0, 2] += self.r_increment
            print('update rotations', '\n', self.rotation.tolist())
                
        else:
            rotation, _ = cv2.Rodrigues(self.rotation)
            if x <= self.left_threshold:         
                translation = np.array([[self.t_increment, 0, 0]], dtype = np.float32)
                self.translation += np.matmul(translation, rotation)
                
            elif self.left_threshold < x < self.right_threshold:
                translation = np.array([[ 0, self.t_increment, 0]], dtype = np.float32)
                self.translation += np.matmul(translation, rotation)
        
            else:
                translation = np.array([[0, 0, self.t_increment]], dtype = np.float32)
                self.translation += np.matmul(translation, rotation)   
                
            print('update translations', '\n', self.translation.tolist())

        
    def down_dial(self, x, y):      
        if y <= self.rt_threshold:
            if x <= self.left_threshold:
                self.rotation[0, 0] -= self.r_increment
            elif self.left_threshold < x < self.right_threshold:
                self.rotation[0, 1] -= self.r_increment
            else:
                self.rotation[0, 2] -= self.r_increment
            print('update rotations', '\n', self.rotation.tolist())
            
        else:
            rotation, _ = cv2.Rodrigues(self.rotation)
            if x <= self.left_threshold:         
                translation = np.array([[self.t_increment, 0, 0]], dtype = np.float32)
                self.translation -= np.matmul(translation, rotation)
                
            elif self.left_threshold < x < self.right_threshold:
                translation = np.array([[ 0, self.t_increment, 0]], dtype = np.float32)
                self.translation -= np.matmul(translation, rotation)
        
            else:
                translation = np.array([[0, 0, self.t_increment]], dtype = np.float32)
                self.translation -= np.matmul(translation, rotation)   
                    
            print('update translations', '\n', self.translation.tolist())

    def reset_dial(self):
        self.rotation = np.zeros((1,3), dtype=np.float32)
        self.translation = -self.reference[0].copy().reshape(1,3)
        print('update vectors', '\n', 
              self.rotation.tolist(), '\n', 
              self.translation.tolist())   

def prep_values(points):
    '''
    First point is the center point. 
    Direction between first and second point is kept.
    The output coordinates are vertices of a right triangle with legs of length 1
    
    '''
    vectors = points - points[0] 
    vectors[2] = np.cross(vectors[1], vectors[2]) 
    normals = np.linalg.norm(vectors[1:], axis = 1).reshape(-1, 1)
    vectors[1:] /= normals
    return vectors + points[0]
    
        
def homogenous_transformation(p, p_prime):
    '''
    Find the unique homogeneous affine transformation that
    maps a set of 3 points to another set of 3 points in 3D
    space:

        p_prime == np.dot(p, R) + t

    where `R` is an unknown rotation matrix, `t` is an unknown
    translation vector, and `p` and `p_prime` are the original
    and transformed set of points stored as row vectors:

        p       = np.array((p1,       p2,       p3))
        p_prime = np.array((p1_prime, p2_prime, p3_prime))

    The result of this function is an augmented 4-by-4
    matrix `A` that represents this affine transformation:

        np.column_stack((p_prime, (1, 1, 1))) == \
            np.dot(np.column_stack((p, (1, 1, 1))), A)

    Source: https://math.stackexchange.com/a/222170 (robjohn)
    '''

    # construct intermediate matrix
    Q       = p[1:]       - p[0]
    Q_prime = p_prime[1:] - p_prime[0]

    # calculate rotation matrix
    R = np.dot(np.linalg.inv(np.vstack((Q, np.cross(*Q)))),
               np.vstack((Q_prime, np.cross(*Q_prime))))

    # calculate translation vector
    t = p_prime[0] - np.dot(p[0], R)
    
    return R, t
#    #calculate affine transformation matrix
#    return np.hstack((np.vstack((R, t)),
#                            (0, 0, 0, 1)))