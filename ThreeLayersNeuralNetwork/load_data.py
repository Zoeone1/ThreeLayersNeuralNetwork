import numpy as np
import pickle

def read_images(file_name):
    with open(file_name, "rb") as f:
        data_dict = pickle.load(f, encoding='latin1')
        images = data_dict['data']
        images = images.reshape((-1, 3, 32, 32))
        images = np.transpose(images, (0, 2, 3, 1))
        images = images.reshape(images.shape[0], -1)
        images = (images / 255.0) - 0.5 #正则化，更快收敛
    return images

def read_labels(file_name):
    with open(file_name, "rb") as f:
        data_dict = pickle.load(f, encoding='latin1')
        labels = data_dict['labels']
        labels = np.array(labels).reshape(-1, 1)
    return labels
