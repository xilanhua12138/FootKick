import argparse
import time
import signal
import cv2
import os
from Arducamlib.Arducam import *
from Arducamlib.ImageConvert import *
import trash.detect as detect

if __name__ == "__main__":

    config_file = "D:\\Desktop\\footkick\\SDVS320_RGB_324x248.cfg"
    verbose = False
    # preview_width = -1
    no_preview = False

    # camera.setCtrl("setFramerate", 2)
    # camera.setCtrl("setExposureTime", 20000)
    # camera.setCtrl("setAnalogueGain", 800)
    # scale_width = preview_width
    lib_path = 'D:\\Desktop\\footkick\\shoe_lib\\'
    shoes_libary = []
    filelist = os.listdir(lib_path)
    for i in range(len(filelist)):
        one_img = cv2.imread(os.path.join(lib_path,filelist[i]))    
        shoes_libary.append(one_img)
    video = cv2.VideoCapture('D:\\Desktop\\footkick\\test.mp4') 
    threshold = 0.7
    ret = True
    prev = None
    prev_isempty = True
    while ret:
        ret, image = video.read()
        # ret, data, cfg = camera.read()
        # display_fps(0)
        if ret:
            # image = convert_image(data, cfg, camera.color_mode)
            imgae = np.array(image[:,:,:3])
            cv2.imshow("Arducam", image) #248,324,4
            cv2.waitKey(10)
            mean_dire = np.array([])
            is_Shoe = detect.checkShoe_one(imgae, shoes_libary,threshold=threshold)
            if is_Shoe:
                cur = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                direct,dire_vec2 = detect.calcDirect_one(image)
                if not prev_isempty:
                    p0 = cv2.goodFeaturesToTrack(prev,40,0.06,10)
                    p1,st,err = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None,winSize=(15,15),maxLevel=2)
                    dire_vec1 = np.array([0,0])
                    for i in range(len(st)):
                        if st[i] == 1:
                            dire_vec1[0] += p0[i,:,0] - p1[i,:,0]
                            dire_vec1[1] += p0[i,:,1] - p1[i,:,1]
                    dire_vec1[0] /= len(np.nonzero(st)[0])
                    dire_vec1[1] /= len(np.nonzero(st)[1])
                    mean_dire = [(0.5*dire_vec1[0]+0.5*dire_vec2[0]), (0.5*dire_vec1[1]+0.5*dire_vec2[1])]

                prev = cur
                prev_isempty = False
                dire = mean_dire if len(mean_dire) !=0 else dire_vec2
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
                print('current direc:',position)
            else:
                print('not shoe')

        key = cv2.waitKey(1)
        if key == ord('q'):
            exit_ = True

