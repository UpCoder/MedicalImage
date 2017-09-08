# -*- coding: utf-8 -*-
# 构建验证集合的数据，使用完整的病灶
import os
from Tools import read_mhd_images, shuffle_image_label


class ValDataSet:
    def __init__(self, data_path, new_size, shuffle=True, phase='ART'):
        self.data_path = data_path
        self.phase = phase
        self.shuffle = True
        self.images, self.labels, self.image_names = ValDataSet.load_data_path(data_path, new_size, self.phase)
        if shuffle:
            self.images, self.labels = shuffle_image_label(self.images, self.labels)

    @staticmethod
    def load_data_path(path, new_size, phase_name):
        casees = os.listdir(path)
        labels = []
        image_pathes = []
        for case_name in casees:
            cur_path = os.path.join(path, case_name, phase_name + '_ROI.mhd')
            cur_label = int(case_name[-1])
            labels.append(cur_label)
            image_pathes.append(cur_path)

        images = read_mhd_images(
            image_pathes,
            new_size=new_size
        )
        return images, labels, image_pathes

    def show_error_name(self, error_index, error_record):
        error_names = []
        for index in error_index:
            error_names.append(self.image_names[index])
        #　error_names.sort()
        for index, error_name in enumerate(error_names):
            print error_name, error_record[index]
