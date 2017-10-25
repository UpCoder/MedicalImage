# -*- coding: utf-8 -*-
# 构建验证集合的数据，使用完整的病灶
import os
from Tools import read_mhd_images, shuffle_image_label, extract_avg_liver_dict, read_mhd_image, save_mhd_image
import shutil
import numpy as np
from ValData import ValDataSet

class ValDataSetMultiPhase:
    def __init__(self, data_path,
                 new_sizes,
                 shuffle=True,
                 label_index_start=0,
                 phases=['NC', 'ART', 'PV'],
                 category_number=5,
                 suffix_names=['_ROI.mhd', '_ROI_Expand.mhd']):
        self.data_path = data_path
        self.phases = phases
        self.valdatasets = []
        self.valdatasets_expand = []
        for phase in self.phases:
            self.valdatasets.append(
                ValDataSet(data_path, new_sizes[0], shuffle, phase, category_number, label_index_start, suffix_name=suffix_names[0])
            )
        for phase in self.phases:
            self.valdatasets_expand.append(
                ValDataSet(data_path, new_sizes[1], shuffle, phase, category_number, suffix_name=suffix_names[1])
            )
        shape = list(np.shape(self.valdatasets[0].images))
        self.images = np.zeros(
            [
                shape[0],
                shape[1],
                shape[2],
                len(self.phases)
            ]
        )
        shape = list(np.shape(self.valdatasets_expand[0].images))
        self.images_expand = np.zeros(
            [
                shape[0],
                shape[1],
                shape[2],
                len(self.phases)
            ]
        )
        self.labels = self.valdatasets[0].labels
        for index, valdataset in enumerate(self.valdatasets):
            self.images[:, :, :, index] = valdataset.images
            print np.shape(self.labels)
            print np.shape(valdataset.labels)
            if np.sum(self.labels != np.array(valdataset.labels)) != 0:
                print 'Error label', index
                print self.labels, valdataset.labels
        for index, valdataset_expand in enumerate(self.valdatasets_expand):
            self.images_expand[:, :, :, index] = valdataset_expand.images

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
                copy_mhd_image(significant, mhd_name)
                print error_name, error_record[index], 'finish copy'
            else:
                print error_name

    def get_next_batch(self, batch_size=None, distribution=None):
        if batch_size is None:
            return self.images, self.images_expand, self.labels
        else:
            if distribution is None:
                random_index = range(len(self.labels))
                np.random.shuffle(random_index)
                batch_index = random_index[:batch_size]
                batch_images = []
                batch_labels = []
                batch_images_expand = []
                for index in batch_index:
                    batch_images.append(
                        self.images[index]
                    )
                    batch_labels.append(
                        self.labels[index]
                    )
                    batch_images_expand.append(
                        self.images_expand[index]
                    )
                return batch_images, batch_images_expand, batch_labels
            else:
                images = []
                labels = []
                images_expand = []
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
                            images_expand.append(self.images_expand[cur_index])
                        if count >= distribution[index]:
                            break
                return images, images_expand, labels
if __name__ == '__main__':
    phase_names = ['NC', 'ART', 'PV']
    val_dataset = ValDataSetMultiPhase(new_sizes=[[128, 128], [64, 64]],
                                       phases=phase_names,
                                       shuffle=False,
                                       category_number=2,
                                       data_path='/home/give/Documents/dataset/MedicalImage/MedicalImage/ROIMultiExpand/train'
                                       )
    print np.shape(val_dataset.images)
    print np.shape(val_dataset.images_expand)
    images, images_expand, label = val_dataset.get_next_batch()
    print np.shape(images), np.shape(images_expand), np.shape(label)