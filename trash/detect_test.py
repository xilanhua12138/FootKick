import numpy as np
import cv2
img = cv2.imread('D:\\Desktop\\footkick\\20221125133639.png')
img_bgr = img
b_ch = img[:,:,0]
r_ch = img[:,:,2]
# 判断红色、蓝色像素点的数量，需要有一定数目再进行运动方向判断
# red_pixels = np.argwhere(cv2.inRange(img_bgr, (0, 0, 245), (0, 0, 255)))
# blue_pixels = np.argwhere(cv2.inRange(img_bgr, (245, 0, 0), (255, 0, 0)))
red_pixels = np.argwhere(r_ch<=50)
blue_pixels = np.argwhere(b_ch<=50)
count_red = len(red_pixels) # 红色像素点的数目
count_blue = len(blue_pixels)
mean_red_pixels = red_pixels.mean(axis = 0) # 左上方为原点，第几行第几列
mean_blue_pixels = blue_pixels.mean(axis = 0)

# 红蓝像素中心位移判断上下还是左右


threshold = 50
if count_red >= threshold and count_blue >= threshold:
    
    mean_red_pixels = red_pixels.mean(axis = 0) # 左上方为原点，第几行第几列
    mean_blue_pixels = blue_pixels.mean(axis = 0)
    # 运动方向：红 → 蓝 左上方为原点，向右为x，向下为y
    cv2.line(img,(int(mean_red_pixels[1]),int(mean_red_pixels[0])),(int(mean_blue_pixels[1]),int(mean_blue_pixels[0])),(0,255,0),3)
    direction_vec = [mean_blue_pixels[1] - mean_red_pixels[1], mean_blue_pixels[0] - mean_red_pixels[0]]