import numpy as np
from global_objects import GlobalConfig as G
from camera_process.SyncFrame import cam_read, fill_array, save_array, pace_maker
from camera_process.SyncRectify4 import array_inference, post_process
from camera_process.Utils import shared_array
from camera_process.read_h5 import read_h5
from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import RawValue
'''
This is the master script for running multiple camera with inference using 
    multiprocessing. Inference cannot be turned off in this script
'''
def master_parallel(recording = True, to_video = False):
    # init variables
    wait_time = 5
    
      
    #output settings
    frame_gap = 1/G.FPS   #0.0002 add this line when print pace and perf_counter() are used
    init_buffer = -int(G.FPS * wait_time)
    
    #shared variables
    count = RawValue('i', init_buffer)
    control = RawValue('i', 1) # start at 1 so that control depended process start only when count >= 0
    title = RawValue('i', 0)  # file will start to save after the first array is completed 
    switch = RawValue('i', 1) 
    lock = Lock()
    
    sync_processes = []
    arrays = []
    data = []
    
    for ID in range(G.cam_num):
        array_0 = shared_array(G.Tshape, np.uint8)
        array_1 = shared_array(G.Tshape, np.uint8)
        
        img_0 = shared_array(G.Ishape, np.uint8)
        img_1 = shared_array(G.Ishape, np.uint8) 
        data_0 = shared_array(G.kp_shape, np.int16)
        data_1 = shared_array(G.kp_shape, np.int16)
        
        process_read = Process(target=cam_read, 
                               args = (ID, count, switch, array_0, array_1))
        process_infer = Process(target=array_inference, 
                               args = (ID, count, control, switch, 
                                       array_0, array_1, 
                                       img_0, img_1, data_0, data_1))
        
        sync_processes = sync_processes + [process_read] + [process_infer]         
        arrays = arrays + [array_0] + [array_1]
        data = data + [img_0] + [img_1] + [data_0] + [data_1]
    
    if recording:    
        central_0 = shared_array(G.Fshape, np.uint8)
        central_1 = shared_array(G.Fshape, np.uint8)
        process = Process(target=fill_array, 
                          args = (count, control, title, switch, 
                                  central_0, central_1, *arrays))
        sync_processes.append(process)   
        process = Process(target=save_array, 
                          args = (count, title, switch, central_0, central_1))
        sync_processes.append(process)
    
    process_post = Process(target=post_process, 
                           args = (count, control, switch, *data))
    process_pace = Process(target=pace_maker, 
                           args = (frame_gap, count, control, 
                                   title, lock, switch))
    sync_processes = sync_processes + [process_post] + [process_pace]
      
    for i in range(len(sync_processes)):
        sync_processes[i].start()
        
    for i in range(len(sync_processes)):
        sync_processes[i].join()
    
    if to_video:
        read_h5()
