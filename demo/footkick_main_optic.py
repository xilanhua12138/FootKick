import argparse
import time
import signal
import cv2
import os
from Arducamlib.Arducam import *
from Arducamlib.ImageConvert import *
import trash.detect as detect
exit_ = False


def sigint_handler(signum, frame):
    global exit_
    exit_ = True


signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)


def display_fps(index):
    display_fps.frame_count += 1

    current = time.time()
    if current - display_fps.start >= 1:
        print("fps: {}".format(display_fps.frame_count))
        display_fps.frame_count = 0
        display_fps.start = current


display_fps.start = time.time()
display_fps.frame_count = 0

if __name__ == "__main__":

    config_file = "/home/yunhaoshui/FootKick/SDVS320_RGB_324x248.cfg"
    verbose = False
    # preview_width = -1
    no_preview = False

    camera = ArducamCamera()

    if not camera.openCamera(config_file):
        raise RuntimeError("Failed to open camera.")

    if verbose:
        camera.dumpDeviceInfo()

    camera.start()
    # camera.setCtrl("setFramerate", 2)
    # camera.setCtrl("setExposureTime", 20000)
    # camera.setCtrl("setAnalogueGain", 800)
    # scale_width = preview_width

    threshold = 0.7
    buffer = []
    ret = True
    prev = None
    prev_isempty = True
    while not exit_:
        ret, data, cfg = camera.read()
        shoes_libary = []
        # display_fps(0)
        if ret:
            image = convert_image(data, cfg, camera.color_mode)
            # image = convert_image(data, cfg, camera.color_mode)
            imgae = np.array(image[:,:,:3])
            # image = preprocess.DBSCAN_denoise(image, 1.4,5)
            cv2.imshow("Arducam", image) #248,324,4
            cv2.waitKey(1)
            dire_vec1 = np.array([])
            is_Shoe = detect.checkShoe_one(imgae, shoes_libary, threshold=threshold)
            if is_Shoe:
                cur = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # direct,dire_vec2 = detect.calcDirect_one(image)
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
                    # mean_dire = [(0.7*dire_vec1[0]+0.3*dire_vec2[0]), (0.7*dire_vec1[1]+0.3*dire_vec2[1])]
                    # mean_dire = dire_vec1
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
                print('current direc:',position)
            else:
                print('not shoe')
        else:
            print("timeout")

        key = cv2.waitKey(1)
        if key == ord('q'):
            exit_ = True
        elif key == ord('s'):
            np.array(data, dtype=np.uint8).tofile("image.raw")

    camera.stop()
    camera.closeCamera()
