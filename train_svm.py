from pathlib import Path
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
import numpy as np
import skimage
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.io import imread
from skimage.transform import resize
import cv2

matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())

def load_image_files(container_path, dimension=(64, 64)):

    """
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset

    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to

    Returns
    -------
    Bunch
    """

    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            if img.shape == (324,248,3):  
                print(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

image_dataset = load_image_files("/home/yunhaoshui/FootKick/clean_dataset/")

X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3,random_state=109
    )

from sklearn.decomposition import PCA
X = X_train
pca = PCA(n_components=1000) #实例化
pca = pca.fit(X)
X_pca_train = pca.transform(X) #获取新矩阵

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid)
clf.fit(X_pca_train, y_train)

X_pca_test = pca.transform(X_test)
y_pred = clf.predict(X_pca_test)

print("Classification report for - \n{}:\n{}\n".format(
    clf, metrics.classification_report(y_test, y_pred)))

import time
pca_time = []
svm_time = []
for i in range(X_test.shape[0]):
    X_i = X_test[i,:].reshape(1,-1)

    pca_start_time = time.time()
    X_pca_test_i = pca.transform(X_i)
    pca_end_time = time.time()
    pca_time.append(pca_end_time-pca_start_time)

    svm_start_time = time.time()
    y_pred = clf.predict(X_pca_test_i)
    svm_end_time = time.time()
    svm_time.append(svm_end_time-svm_start_time)

print('pca time:',np.mean(pca_time)*1000,'ms')
print('svm time:',np.mean(svm_time)*1000,'ms')