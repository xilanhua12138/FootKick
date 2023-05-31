from utils import svm_inference
import time
import cv2
model = svm_inference.build_SVM('/home/yunhaoshui/FootKick/resources/svm_model.pkl')
img = cv2.imread('/home/yunhaoshui/FootKick/clean_dataset/shoe/20221207_154913_Image_016.bmp')
img_resize = cv2.resize(img,(64,64))
cv2.imshow('',img_resize)
cv2.waitKey(0)
start_time = time.time()
res = svm_inference.inference_with_SVM(img, (64,64), model)

end_time = time.time()

proccestime = end_time - start_time

print('pred:',res ,'process time:', proccestime*1000,'ms')