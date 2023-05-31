import pickle
from skimage.io import imread
from skimage.transform import resize
import numpy as np

def build_SVM(model_path):
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    return clf

def inference_with_SVM(image, dimension, clf_model):

    img_resized = resize(image, dimension, anti_aliasing=True, mode='reflect')
    flat_data = img_resized.flatten()
    flat_data = np.array(flat_data)
    flat_data = flat_data.reshape(1,-1)
    class_pred = clf_model.predict(flat_data)
    return class_pred