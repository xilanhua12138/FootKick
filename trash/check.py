from __future__ import print_function
import cv2
import numpy as np
import os 
def alignImages(im1, im2): #对齐图像
    MAX_FEATURES = 600
    GOOD_MATCH_PERCENT = 0.15

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
      points1[i, :] = keypoints1[match.queryIdx].pt
      points2[i, :] = keypoints2[match.trainIdx].pt

      # Find homography
      h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

      # Use homography
      height, width, channels = im2.shape
      im1Reg = cv2.warpPerspective(im1, h, (width, height))


    return im1Reg, h

def crossCorrelation(img1Reg,img2): #计算对齐后的图像与图库中图像的相关系数
    img1=img1Reg
    cor_coefficient=np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2))
    cor_coefficient=np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2))
    # if cor_coefficient >= 0.8:
    #     return 1
    # else:
    #     return 0
    return cor_coefficient

def compareImages(img1,imgSet):  #比对图库中每一张图像后，判断目标是否为鞋子，输入为当前图像和鞋子图像库
    result = []
    #convexhull and fillit
    gray = cv2.cvtColor(img1,cv2.COLOR_BGRA2GRAY)   #转化为灰度图
    ret,binary= cv2.threshold(gray,127,255,0)     #转化为二值图
    points = np.where(binary != 255)
    points = np.array([points[1],points[0]]).T
    hull= cv2.convexHull(points)     #寻找凸包并绘制
    calc_img = 255*np.ones_like(img1[:,:,:3])
    cv2.polylines(calc_img,[hull],True,(0,0,0),3)

    for i in range(len(imgSet)):
        # im1Reg, h=alignImages(img1,imgSet[i])
        result.append(crossCorrelation(calc_img,imgSet[i]))
    return max(result)

if __name__=='__main__':
    lib_path = 'D:\\Desktop\\footkick\\shoe_lib\\'
    shoes_libary = []
    filelist = os.listdir(lib_path)
    for i in range(len(filelist)):
        one_img = cv2.imread(os.path.join(lib_path,filelist[i]))    
        shoes_libary.append(one_img)
    img = cv2.imread('D:\\Desktop\\footkick\\shoe_lib\\1.png')
    print(compareImages(img, shoes_libary))