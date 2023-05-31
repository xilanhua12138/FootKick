import argparse
import time
import signal
import cv2
import os
import torch
from Arducamlib.Arducam import *
from Arducamlib.ImageConvert import *
from yolov5 import buildmodel_engine, buildmodel_onnx, inference
def run(
    config_path, 
    weight_path, 
    model_type='engine', 
    data_path = None,
    device=torch.device('cuda:0'), 
    half=True, 
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

    # Initialize Yolov5
    model = inference.model_init(weight_path, model_type, data_path, device, half=True, imgsz=320)

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
            dire_tmp = np.array([])
            position = None

            if len(cls) == 0:
                print('not shoe')
                # cv2.putText(img0, 'no shoe', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            if len(cls) != 0:
                if cls[0] == 0:
                    print('not shoe')
                    # cv2.putText(img0, 'not shoe', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                if cls[0] == 1:
                    start_time = time.time()
                    # cur_xyxy = xyxy.cpu().numpy()
                    cur_center = [(xyxy[0][0].cpu()+xyxy[0][2].cpu())/2, (xyxy[0][1].cpu()+xyxy[0][3].cpu())/2]
                    # direct,dire_vec2 = detect.calcDirect_one(image)
                    dire_tmp = np.array([0,0])
                    if not prev_isempty:
                        dire_tmp[0] = cur_center[0] - prev_center[0]
                        dire_tmp[1] = cur_center[1] - prev_center[1]
                    prev_center = cur_center
                    prev_isempty = False
                    dire = dire_tmp if len(dire_tmp) !=0 else [0,0]
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

            cv2.putText(img0, str(position), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            cv2.imshow("Arducam", img0) #248,324,4
            cv2.waitKey(10)
        else:
            print('average process time:', np.average(total_time),'ms')
            return

if __name__ == "__main__":
    config_path = "/home/yunhaoshui/FootKick/resources/SDVS320_RGB_324x248.cfg"
    weight_path = "/home/yunhaoshui/FootKick/resources/best.engine"
    data_path = '/home/yunhaoshui/FootKick/resources/footkick.yaml'
    run(config_path=config_path, weight_path=weight_path, data_path =data_path)