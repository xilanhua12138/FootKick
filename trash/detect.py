import trash.check as check
import numpy as np
import cv2


def checkShoe(buffer,shoe_libary, threshold):
    max_corrs = []
    for img in buffer:
        corr_lib = check.compareImages(img, shoe_libary)
        max_corrs.append(corr_lib)
    if max(max_corrs) > threshold:
        return True
    else:
        return False

def checkShoe_one(img,shoe_libary, threshold):
    b_ch = img[:,:,0]
    r_ch = img[:,:,2]
    total = len(np.argwhere(b_ch<=50)) + len(np.argwhere(r_ch<=50))
    if total > 200:
        return True
    else:
        return False

def calcDirect(buffer):
    directions = []   
    for img in buffer:
        img_bgr = img
        ccc = 1
        b_ch = img[:,:,0]
        r_ch = img[:,:,2]
        # 判断红色、蓝色像素点的数量，需要有一定数目再进行运动方向判断
        red_pixels = np.argwhere(r_ch<=50)
        blue_pixels = np.argwhere(b_ch<=50)
        # red_pixels = np.argwhere(cv2.inRange(img_bgr, (0, 0, 245), (0, 0, 255)))
        # blue_pixels = np.argwhere(cv2.inRange(img_bgr, (245, 0, 0), (255, 0, 0)))
        count_red = len(red_pixels) # 红色像素点的数目
        count_blue = len(blue_pixels)
        # 红蓝像素中心位移判断上下还是左右
        threshold = 50
        if count_red >= threshold and count_blue >= threshold:
            
            mean_red_pixels = red_pixels.mean(axis = 0) # 左上方为原点，第几行第几列
            mean_blue_pixels = blue_pixels.mean(axis = 0)
            # 运动方向：红 → 蓝 左上方为原点，向右为x，向下为y
            # cv2.line(img,(int(mean_red_pixels[1]),int(mean_red_pixels[0])),(int(mean_blue_pixels[1]),int(mean_blue_pixels[0])),(0,255,0),3)
            direction_vec = [mean_blue_pixels[1] - mean_red_pixels[1], mean_blue_pixels[0] - mean_red_pixels[0]]
        else:
            direction_vec = []
        if direction_vec != []:
            if abs(direction_vec[0]) > abs(direction_vec[1]):
                if direction_vec[0] > 0:
                    position = 'right'
                else:
                    position = 'left'
            else:
                if direction_vec[1] > 0:
                    position = 'down'
                else:
                    position = 'up'
        else:
            position = ''
        directions.append(position)
    return directions

def calcDirect_one(img):
    img_bgr = img
    b_ch = img[:,:,0]
    r_ch = img[:,:,2]
    # 判断红色、蓝色像素点的数量，需要有一定数目再进行运动方向判断
    red_pixels = np.argwhere(r_ch<=50)
    blue_pixels = np.argwhere(b_ch<=50)
    count_red = len(red_pixels) # 红色像素点的数目
    count_blue = len(blue_pixels)
    
    # 红蓝像素中心位移判断上下还是左右
    
    
    threshold = 50
    if count_red >= threshold and count_blue >= threshold:
        
        mean_red_pixels = red_pixels.mean(axis = 0) # 左上方为原点，第几行第几列
        mean_blue_pixels = blue_pixels.mean(axis = 0)
        # 运动方向：红 → 蓝 左上方为原点，向右为x，向下为y
        direction_vec = [mean_blue_pixels[1] - mean_red_pixels[1], mean_blue_pixels[0] - mean_red_pixels[0]]
    else:
        direction_vec = [0,0]
        
    if direction_vec != [0,0]:
        if abs(direction_vec[0]) > abs(direction_vec[1]):
            if direction_vec[0] > 0:
                position = 'right'
            else:
                position = 'left'
        else:
            if direction_vec[1] > 0:
                position = 'down'
            else:
                position = 'up'
    else:
        position = ''
    return position,direction_vec