import argparse
import time
import signal
import cv2
import os
import torch
from Arducamlib.Arducam import *
from Arducamlib.ImageConvert import *
from classification import efficient_model


def judge_from_window(pos_list):
    count = {}
    for i in set(pos_list):
        count[i] = pos_list.count(i)
    
    max_direction = max(count, key=count.get)
    return max_direction

def is_mostly_white(img, threshold):
    # 对图片进行灰度化，将三通道变成单通道
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算图片中所有像素的总和
    total = np.sum(img)

    # 计算图片中所有元素的个数
    count = img.size

    # 计算平均值，即每个元素的平均像素值
    mean = total / count

    # 如果平均值大于阈值，则返回True，否则返回False
    return mean > threshold

def run(
    config_path, 
    weight_path, 
    model_type='onnx', 
    data_path = None,
    device='cpu', 
    half=False, 
    imgsz=32
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
    model = efficient_model.build_model(weight_path, device)
    total_time = []
    cls_time = []
    LK_time = []
    dis = cv2.DISOpticalFlow_create(1)
    # Begin detection
    while ret:
        ret, image = video.read()
        
        frame_count=video.get(cv2.CAP_PROP_FRAME_COUNT)
        # ret, data, cfg = camera.read()
        # display_fps(0)
        if ret:
            # image = convert_image(data, cfg, camera.color_mode)
            image = np.array(image[:,:,:3])
            image = cv2.resize(image,(32,32))
            start_time0 = time.time()
            result = efficient_model.inference(model, image, imgsz)
            cls = np.argmax(result)
            end_time0 = time.time()
            clstime = end_time0 - start_time0
            # print('yolo time:',round((yolotime)*1000,2),'ms')
            # image = preprocess.DBSCAN_denoise(image, 1.4,5)
            
            dire_vec1 = np.array([])
            position = None
            
            white = is_mostly_white(image,245)

            if white:
                cv2.putText(image, 'nothing', (200,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
                print('nothing')
                position = None

            if not white:

                if cls == 0: # means not shoe
                    cv2.putText(image, 'unshoe', (200,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
                    print('not shoe')
                    position = None

                if cls == 1: # means there exists a shoe
                    cv2.putText(image, 'shoe', (200,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
                    start_time = time.time()
                    cur = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    cur = cv2.resize(cur, (32,32))
                    if not prev_isempty:
                        flow = dis.calc(prev, cur, None)
                    else:
                        flow = np.zeros_like(image)

                    prev = cur
                    prev_isempty = False
                    mean_x = np.mean(flow[..., 0])
                    mean_y = np.mean(flow[..., 1])

                    if abs(mean_x) > abs(mean_y):
                        if mean_y > 0:
                            position = 'right'
                        else:
                            position = 'left'
                    else:
                        if mean_y > 0:
                            position = 'up'
                        else:
                            position = 'down'
                    
                    end_time = time.time()
                    LKtime = end_time-start_time

                    # print('current direc:',position,'LK process time:', round((LK_time)*1000,2),'ms')
                    total_time.append(round((clstime+LKtime)*1000,2))
                    cls_time.append(round((clstime)*1000,2))
                    LK_time.append(round((LKtime)*1000,2))
                    print(
                        'total_time', round((clstime+LKtime)*1000,2),'ms', 
                        ' clstime:', round((clstime)*1000,2),'ms',
                        ' DIS time:',round((LKtime)*1000,2),'ms'
                        )
            
            window.append(position)
            if len(window) == window_size:
                current_window = list(window)
                action = judge_from_window(current_window)
            cv2.putText(image, str(action), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            cv2.imshow("Arducam", image) #248,324,4
            cv2.waitKey(20)
        
        else:
            print('average process time:', np.average(total_time),'ms',
                  'average cls time:', np.average(cls_time),'ms',
                  'average DIS time:', np.average(LK_time),'ms')
            return

if __name__ == "__main__":
    config_path = "/home/yunhaoshui/FootKick/resources/SDVS320_RGB_324x248.cfg"
    weight_path = "/home/yunhaoshui/FootKick/resources/efficientnet_imgsz32.onnx"
    data_path = '/home/yunhaoshui/FootKick/resources/footkick_openmmlab.yaml'
    run(config_path=config_path, weight_path=weight_path, data_path =data_path)