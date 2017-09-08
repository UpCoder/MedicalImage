# -*- coding: utf-8 -*-
# 构建验证集合的数据，使用完整的病灶
import os
from Tools import read_mhd_images, shuffle_image_label, extract_avg_liver_dict


class ValDataSet:
    def __init__(self, data_path, new_size, shuffle=True, phase='ART'):
        self.data_path = data_path
        self.phase = phase
        self.shuffle = True
        self.avg_liver_dict = extract_avg_liver_dict()

        self.images, self.labels, self.image_names = ValDataSet.load_data_path(data_path, new_size, self.phase, self.avg_liver_dict)
        if shuffle:
            self.images, self.labels = shuffle_image_label(self.images, self.labels)

    @staticmethod
    def load_data_path(path, new_size, phase_name, avg_liver_dict):
        def get_phase_index(phase):
            if phase == 'NC':
                return 0
            if phase == 'ART':
                return 1
            if phase == 'PV':
                return 2
        casees = os.listdir(path)
        labels = []
        image_pathes = []
        avg_liver_values = []
        for case_name in casees:
            srrid = int(case_name.split('_')[0])
            phase_index = get_phase_index(phase=phase_name)
            avg_liver_value = avg_liver_dict[srrid][phase_index]
            avg_liver_values.append(avg_liver_value)
            cur_path = os.path.join(path, case_name, phase_name + '_ROI.mhd')
            cur_label = int(case_name[-1])
            labels.append(cur_label)
            image_pathes.append(cur_path)

        images = read_mhd_images(
            image_pathes,
            new_size=new_size,
            avg_liver_values=avg_liver_values
        )
        return images, labels, image_pathes

    def show_error_name(self, error_index, error_record):
        error_names = []
        for index in error_index:
            error_names.append(self.image_names[index])
        #　error_names.sort()
        for index, error_name in enumerate(error_names):
            print error_name, error_record[index]
