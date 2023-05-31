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

def Kalman(x_prev, x_next_ob, P_prev, H, Q, R, F):
    """
    x: [x, y, dx, dy].T
    u: [0]
    F: (4,4)
    P: (4,4)
    Q: (4,4)
    H: (4,4)
    """
    # Predict
    x_next_bar = F @ x_prev.T 
    P_next_bar = F @ P_prev @ F.T + Q

    # Observe
    z_next = H @ x_next_ob

    # Update
    K_next = P_next_bar @ H.T @ np.linalg.inv((H @ P_next_bar @ H.T + R))

    x_next = x_next_bar + K_next @ (z_next - H @ x_next_bar)

    P_next = (np.eye(4) - K_next @ H) @ P_next_bar

    return x_next, P_next


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
    list_point = []
    # Openvideo
    video = cv2.VideoCapture('/home/yunhaoshui/FootKick/test.mp4') 
    conf_threshold = 0.7
    ret = True
    prev_isempty = True
    action = None
    window_size_action = 1
    window_size_point = 5
    from collections import deque
    window = deque(maxlen=window_size_action)
    window_b = deque(maxlen=window_size_point)
    window_r = deque(maxlen=window_size_point)
    # Initialize efficientnet
    model = efficient_model.build_model(weight_path, device)
    total_time = []
    cls_time = []
    LK_time = []

    # Set Kalman filter
    F = np.array(
        [[1,0,0,1],
         [0,1,0,1],
         [0,0,1,0],
         [0,0,0,1]]
    )

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    R = np.eye(2) *0.1
    Q = np.array([[0.01, 0, 0, 0], [0, 0.01, 0, 0], [0, 0, 0.01, 0], [0, 0, 0, 0.01]])

    P_prev_b = np.array(
        [[1,0,0,0],
         [0,1,0,0],
         [0,0,1,0],
         [0,0,0,1]]
    )

    P_prev_r = np.array(
        [[1,0,0,0],
         [0,1,0,0],
         [0,0,1,0],
         [0,0,0,1]]
    )

    # Begin detection
    while ret:
        ret, image = video.read()
        
        frame_count=video.get(cv2.CAP_PROP_FRAME_COUNT)
        # ret, data, cfg = camera.read()
        # display_fps(0)
        if ret:
            # image = convert_image(data, cfg, camera.color_mode)
            image = np.array(image[:,:,:3])
            image0 = np.array(image)
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
                cv2.putText(image0, 'nothing', (200,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
                print('nothing')
                position = None

            if not white:

                if cls == 0: # means not shoe
                    cv2.putText(image0, 'unshoe', (200,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
                    print('not shoe')
                    position = None

                if cls == 1: # means there exists a shoe
                    cv2.putText(image0, 'shoe', (200,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
                    start_time = time.time()
                    cur = cv2.resize(image,(32,32))
                    cur = cv2.cvtColor(cur,cv2.COLOR_BGR2GRAY)
                    cur_point = np.argwhere(cur<240)          

                    center_b = cur_point.mean(axis=0) #(y,x)
                    
                    if prev_isempty :
                        center_b_prev = center_b

                        x_prev_b = np.append(center_b,[0,0],axis=0)

                        prev_isempty = False

                    else:
                        list_point.append(center_b)
                        dxdy_blue = center_b - center_b_prev


                        x_next_ob_b = np.append(center_b,dxdy_blue,axis=0)
 

                        x_next_b, P_next_b = Kalman(x_prev_b, x_next_ob_b,  P_prev_b, H, Q, R, F)

                        
                        cv2.circle(image0, (int(x_next_b[1]*324/32), int(x_next_b[0]*248/32)), 10, (0, 0, 0), -1)
                        # cv2.circle(image0, (int(center_b[1]*324/16), int(center_b[0]*248/16)), 10, (0, 255, 0), -1)
                    
                        VEC = x_next_b[:2]-x_prev_b[:2]
                        print(VEC)
                        dire_vec1 = np.copy(VEC)
                        dire_vec1[0] = VEC[1]
                        dire_vec1[1] = VEC[0]
                        position = None
                        dire = dire_vec1 if len(dire_vec1) !=0 else [0,0]
                        if abs(dire[0]) > abs(dire[1]):
                            if dire[0] > 0:
                                position = 'right'
                            if dire[0] <= 0:
                                position = 'left'
                        if abs(dire[0]) <= abs(dire[1]):
                            if dire[1] > 0:
                                position = 'down'
                            if dire[1] <= 0: 
                                position = 'up'
                        
                        end_time = time.time()
                        LKtime = end_time-start_time

                        x_prev_b = x_next_b

                        P_prev_b = P_next_b

                        center_b_prev = center_b

                    
                    
                    end_time = time.time()
                    LKtime = end_time-start_time

                    # print('current direc:',position,'LK process time:', round((LK_time)*1000,2),'ms')
                    total_time.append(round((clstime+LKtime)*1000,2))
                    cls_time.append(round((clstime)*1000,2))
                    LK_time.append(round((LKtime)*1000,2))
                    print(
                        'total_time', round((clstime+LKtime)*1000,2),'ms', 
                        ' clstime:', round((clstime)*1000,2),'ms',
                        ' direction time:',round((LKtime)*1000,2),'ms'
                        )
            np.save('list_point.npy',list_point)
            window.append(position)
            if len(window) == window_size_action:
                current_window = list(window)
                action = judge_from_window(current_window)
            cv2.putText(image0, str(action), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            cv2.imshow("Arducam", image0) #248,324,4
            cv2.waitKey(20)
        
        else:
            print('average process time:', np.average(total_time),'ms',
                  'average cls time:', np.average(cls_time),'ms',
                  'average direction time:', np.average(LK_time),'ms')
            return

if __name__ == "__main__":
    config_path = "/home/yunhaoshui/FootKick/resources/SDVS320_RGB_324x248.cfg"
    weight_path = "/home/yunhaoshui/FootKick/resources/efficientnet_imgsz32.onnx"
    data_path = '/home/yunhaoshui/FootKick/resources/footkick_openmmlab.yaml'
    run(config_path=config_path, weight_path=weight_path, data_path =data_path)