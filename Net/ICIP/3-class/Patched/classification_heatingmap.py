# -*- coding=utf-8 -*-
import numpy as np
from PIL import Image
import os
import scipy.io as scio
from Tools import calculate_acc_error
class_num = 5
def get_label_from_pixelvalue(pixel_value):
    if pixel_value[0] >= 200 and pixel_value[1] >= 200 and pixel_value[2] >= 200:
        return 4
    if pixel_value[1] >= 200 and pixel_value[2] >= 200:
        return 3
    if pixel_value[0] >= 200:
        return 2
    if pixel_value[1] >= 200:
        return 0
    if pixel_value[2] >= 200:
        return 1

def generate_feature_by_heatingmap(image):
    '''
    产生一幅热力图对应的特征向量
    :param image:　热力图
    :return:对应的特征向量
    '''
    features = np.zeros([1, class_num], np.float32)
    shape = list(np.shape(image))
    for i in range(shape[0]):
        for j in range(shape[1]):
            pixel_value = image[i, j]
            index = get_label_from_pixelvalue(pixel_value)
            features[0, index] += 1
    features /= np.sum(features)
    return np.array(features).squeeze()

def generate_features_multiheatingmap(dir_path):
    names = os.listdir(dir_path)
    image_paths = [os.path.join(dir_path, name) for name in names]
    features = [generate_feature_by_heatingmap(np.array(Image.open(path))) for path in image_paths]
    return features

def generate_features_labels(data_dir):
    train_features = []
    train_labels = []
    val_features = []
    val_labels = []
    for subclass in ['train', 'test']:
        for type in [0, 1, 2]:
            cur_features = generate_features_multiheatingmap(os.path.join(data_dir, subclass, str(type)))
            if subclass == 'train':
                train_features.extend(cur_features)
                train_labels.extend([type] * len(cur_features))
            else:
                val_features.extend(cur_features)
                val_labels.extend([type] * len(cur_features))
    scio.savemat('data.mat', {
        'train_features': train_features,
        'train_labels': train_labels,
        'val_features': val_features,
        'val_labels': val_labels
    })
    return train_features, train_labels, val_features, val_labels

if __name__ == '__main__':
    train_features, train_labels, val_features, val_labels = \
        generate_features_labels('/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/heatingmap/ICIP/3-classes/only_patch')
    print np.shape(train_features), np.shape(train_labels), np.shape(val_features), np.shape(val_labels)
    from BoVW.classification import SVM, LinearSVM, KNN
    #　predicted_label = SVM.do(train_features, train_labels, val_features, val_labels, adjust_parameters=True)
    predicted_label, c_params, g_params, accs = SVM.do(train_features, train_labels, val_features, val_labels, adjust_parameters=True)
    print predicted_label
    calculate_acc_error(predicted_label, val_labels)

    # from tools.plot3D import draw_3d
    # draw_3d(c_params, g_params, accs, 'log c', 'log g', '%')