# -*- coding=utf-8 -*-
import numpy as np


'''
    将一段文本转化为向量,不区分大小写，所以每个字母的维度是10+26
    :param txt文本
    :return 对应的向量
'''
def txt2vec(txt):
    # the max length of singel vector
    MAX_SINGLE_LENGTH = 36
    vec = np.zeros(MAX_SINGLE_LENGTH*len(txt), np.uint8)

    for index, c in enumerate(txt):
        single_vec = np.zeros(MAX_SINGLE_LENGTH, np.uint8)
        if '0' <= c <= '9':
            pos = ord(c)-ord('0')
            single_vec[pos] = 1
        else:
            c = (''+c).lower()
            pos = ord(c) - ord('a') + 10
            single_vec[pos] = 1
        vec[index*MAX_SINGLE_LENGTH: (index+1)*MAX_SINGLE_LENGTH] = single_vec
    return vec


'''
    读取ｍｈｄ文件
'''
def read_mhd_image(file_path):
    import SimpleITK as itk
    header = itk.ReadImage(file_path)
    image = itk.GetArrayFromImage(header)
    return np.array(image)


'''
    一次读取多个ｍｈｄ文件
'''
def read_mhd_images(paths, avg_liver_values=None):
    images = []
    for index, path in enumerate(paths):
        # print path
        cur_image = read_mhd_image(path)
        cur_img = np.asarray(cur_image, np.float32)
        if avg_liver_values is not None:
            for i in range(1):
                cur_img = cur_img * cur_img
                cur_img = cur_img / avg_liver_values[index]
        images.append(cur_image)
    return images