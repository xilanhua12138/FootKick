import argparse
import time
import signal
import cv2
import os
import torch
from Arducamlib.Arducam import *
from Arducamlib.ImageConvert import *
from yolov5 import buildmodel_engine, buildmodel_onnx, inference

def judge_from_window(pos_list):
    count = {}
    for i in set(pos_list):
        count[i] = pos_list.count(i)
    
    max_direction = max(count, key=count.get)
    return max_direction


def run(
    config_path, 
    weight_path, 
    model_type='onnx', 
    data_path = None,
    device=torch.device('cpu'), 
    half=False, 
    imgsz=320
    ):

    # Set camera config 
    config_file = config_path
    verbose = False
    # preview_width = -1
    no_preview = False

    # Openvideo
    video = cv2.VideoCapture('/home/yunhaoshui/FootKick/test.mp4') 
    conf_threshold = 0.7
    ret = True
    prev = None
    prev_isempty = True
    action = None
    window_size = 10
    
    from collections import deque
    window = deque(maxlen=window_size)
    # Initialize Yolov5
    model = inference.model_init(weight_path, model_type, data_path, device, half=half, imgsz=320)

    # Begin detection
    while ret:
        ret, image = video.read()
        
        frame_count=video.get(cv2.CAP_PROP_FRAME_COUNT)
        # ret, data, cfg = camera.read()
        # display_fps(0)
        if ret:
            # image = convert_image(data, cfg, camera.color_mode)
            imgae = np.array(image[:,:,:3])
            start_time0 = time.time()
            xyxy,conf,cls,img0 = inference.inference(image, model)
            end_time0 = time.time()
            yolotime = end_time0 - start_time0
            # print('yolo time:',round((yolotime)*1000,2),'ms')
            # image = preprocess.DBSCAN_denoise(image, 1.4,5)
            total_time = []
            dire_vec1 = np.array([])
            position = None

            if len(cls) == 0:
                print('nothing')
                position = None

            if len(cls) != 0:

                if cls[0] == 0: # means not shoe
                    print('not shoe')
                    position = None

                if cls[0] == 1: # means there exists a shoe
                    start_time = time.time()
                    cur = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    if not prev_isempty:
                        p0 = cv2.goodFeaturesToTrack(prev,40,0.06,10)
                        p1,st,err = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(30,30), maxLevel=2)
                        len_valid = len(np.nonzero(st)[0])
                        if len_valid == 0:
                            continue
                        dire_vec1 = np.array([0,0])
                        for i in range(len(st)):
                            if st[i] == 1:
                                dire_vec1[0] += p0[i,:,0] - p1[i,:,0]
                                dire_vec1[1] += p0[i,:,1] - p1[i,:,1]
                        dire_vec1[0] /= len_valid
                        dire_vec1[1] /= len(np.nonzero(st)[1])

                    prev = cur
                    prev_isempty = False
                    dire = dire_vec1 if len(dire_vec1) !=0 else [0,0]
                    if abs(dire[0]) > abs(dire[1]):
                        if dire[0] > 0:
                            position = 'right'
                        else:
                            position = 'left'
                    else:
                        if dire[1] > 0:
                            position = 'down'
                        else:
                            position = 'up'
                    
                    end_time = time.time()
                    LK_time = end_time-start_time

                    # print('current direc:',position,'LK process time:', round((LK_time)*1000,2),'ms')
                    total_time.append(round((yolotime+LK_time)*1000,2))

                    print(
                        'total_time', round((yolotime+LK_time)*1000,2),'ms', 
                        ' yolotime:', round((yolotime)*1000,2),'ms',
                        ' LK time:',round((LK_time)*1000,2),'ms'
                        )
            
            window.append(position)
            if len(window) == window_size:
                current_window = list(window)
                action = judge_from_window(current_window)
            cv2.putText(img0, str(action), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            cv2.imshow("Arducam", img0) #248,324,4
            cv2.waitKey(20)
        
        else:
            print('average process time:', np.average(total_time),'ms')
            return

if __name__ == "__main__":
    config_path = "/home/yunhaoshui/FootKick/resources/SDVS320_RGB_324x248.cfg"
    weight_path = "/home/yunhaoshui/FootKick/resources/best_cpu.onnx"
    data_path = '/home/yunhaoshui/FootKick/resources/footkick.yaml'
    run(config_path=config_path, weight_path=weight_path, data_path =data_path)