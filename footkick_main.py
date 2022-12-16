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

    config_file = "D:\\Desktop\\footkick\\SDVS320_RGB_324x248.cfg"
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
    lib_path = 'D:\\Desktop\\footkick\\shoe_lib\\'
    shoes_libary = []
    filelist = os.listdir(lib_path)
    for i in range(len(filelist)):
        one_img = cv2.imread(os.path.join(lib_path,filelist[i]))    
        shoes_libary.append(one_img)

    threshold = 0.7
    buffer = []
    while not exit_:

        ret, data, cfg = camera.read()
        # display_fps(0)
        if ret:
            cv2.imshow("Arducam", image) #248,324,4
            if len(buffer) == 10:
                del(buffer[0])
            image = convert_image(data, cfg, camera.color_mode)
            imgae = np.array(image[:,:,:3])
            buffer.append(image)
            if len(buffer) == 10:
                is_Shoe = detect.checkShoe(buffer, shoes_libary,threshold=threshold)
                if is_Shoe:
                    directs = detect.calcDirect(buffer)
                    from collections import Counter
                    collection = Counter(directs)
                    most_direct = collection.most_common(1) 
                    cur_direct = detect.calcDirect(image)

                    if most_direct == cur_direct:
                        print('current direction is:',cur_direct)
                    else:
                        collection = Counter(directs[-3:-1])
                        neareast = collection.most_common(1) 
                        print('current direction is:',neareast)
                else:
                    print('not shoe')
            else:
                continue
        else:
            print("timeout")

        key = cv2.waitKey(1)
        if key == ord('q'):
            exit_ = True
        elif key == ord('s'):
            np.array(data, dtype=np.uint8).tofile("image.raw")

    camera.stop()
    camera.closeCamera()
