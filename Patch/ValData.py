# -*- coding: utf-8 -*-
# 构建验证集合的数据，使用完整的病灶
import os
from Tools import read_mhd_images, shuffle_image_label, extract_avg_liver_dict, read_mhd_image, save_mhd_image
import shutil
import numpy as np


class ValDataSet:
    def __init__(self, data_path, new_size, shuffle=True, phase='ART', category_number=5):
        self.data_path = data_path
        self.phase = phase
        self.shuffle = True
        self.avg_liver_dict = extract_avg_liver_dict()
        self.category_number = category_number
        self.images, self.labels, self.image_names = ValDataSet.load_data_path(data_path, new_size, self.phase,
                                                                               self.avg_liver_dict,
                                                                               self.category_number)
        if shuffle:
            self.images, self.labels = shuffle_image_label(self.images, self.labels)

    def get_next_batch(self, batch_size=None, distribution=None):
        if batch_size is None:
            return self.images, self.labels
        else:
            if distribution is None:
                random_index = range(len(self.labels))
                np.random.shuffle(random_index)
                batch_index = random_index[:batch_size]
                batch_images = []
                batch_labels = []
                for index in batch_index:
                    batch_images.append(
                        self.images[index]
                    )
                    batch_labels.append(
                        self.labels[index]
                    )
                batch_images, batch_labels = shuffle_image_label(batch_images, batch_labels)
                return batch_images, batch_labels
            else:
                images = []
                labels = []
                for index, count in enumerate(distribution):
                    cur_indexs = (np.array(self.labels) == index)
                    random_index = range(len(self.labels))
                    np.random.shuffle(random_index)
                    count = 0
                    for cur_index in random_index:
                        if cur_indexs[cur_index]:
                            count += 1
                            images.append(self.images[cur_index])
                            labels.append(self.labels[cur_index])
                        if count >= distribution[index]:
                            break
                images, labels = shuffle_image_label(images, labels)
                return images, labels

    @staticmethod
    def load_data_path(path, new_size, phase_name, avg_liver_dict, category_number):
        def get_phase_index(phase):
            if phase == 'NC':
                return 0
            if phase == 'ART':
                return 1
            if phase == 'PV':
                return 2
        casees = list(os.listdir(path))
        labels = []
        image_pathes = []
        avg_liver_values = []
        casees.sort()
        for case_name in casees:
            # print case_name
            print 'case name is ', case_name
            print case_name.split('_')[0]
            srrid = int(case_name.split('_')[0])
            phase_index = get_phase_index(phase=phase_name)
            avg_liver_value = avg_liver_dict[srrid][phase_index]
            avg_liver_values.append(avg_liver_value)
            cur_path = os.path.join(path, case_name, phase_name + '_ROI.mhd')
            print cur_path
            # print case_name
            cur_label = int(case_name[-1])
            if category_number == 2:
                if cur_label == 0 or cur_label == 1 or cur_label == 3:
                    cur_label = 0
                else:
                    if cur_label == 2 or cur_label == 4:
                        cur_label = 1
            labels.append(cur_label)
            image_pathes.append(cur_path)

        images = read_mhd_images(
            image_pathes,
            new_size=new_size,
            avg_liver_values=avg_liver_values
        )
        return images, labels, image_pathes

    def show_error_name(self, error_index, error_record,
                        error_save_dir='/home/give/Documents/dataset/MedicalImage/MedicalImage/ROI/val_error',
                        copy=True):
        error_names = []
        for index in error_index:
            error_names.append(self.image_names[index])
        #　error_names.sort()
        for index, error_name in enumerate(error_names):
            base_name = os.path.join(
                error_save_dir,
                os.path.basename(os.path.dirname(error_name)) + '_' +
                os.path.basename(error_name).split('.')[0] + '_' + str(error_record[index])
            )
            mhd_name = base_name+'.mhd'
            raw_name = base_name+'.raw'
            if copy:
                if os.path.exists(mhd_name) and os.path.join(raw_name):
                    print error_name, error_record[index], 'exists'
                    continue

                def copy_mhd_image(mhd_path, save_path):
                    mhd_image = read_mhd_image(mhd_path)
                    save_mhd_image(
                        mhd_image,
                        save_path
                    )
                srrid = os.path.basename(os.path.dirname(error_name))
                cur_type = srrid[-1]
                significant = os.path.join(
                    '/home/give/Documents/dataset/MedicalImage/MedicalImage/SignificantLayers/val',
                    cur_type,
                    srrid,
                    'ART_Image.mhd'
                )
                # signficant = error_name.replace('ROI', 'SignificantLayers')
                copy_mhd_image(significant, mhd_name)
                print error_name, error_record[index], 'finish copy'
            else:
                print error_name, error_record[index]
