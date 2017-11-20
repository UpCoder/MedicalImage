# -*- coding=utf-8 -*-
import os
from Config import Config
from glob import glob
import numpy as np
from Tools import shuffle_image_label, read_mhd_image, get_boundingbox, resize_image, show_image, cal_liver_average, compress22dim
from copy import copy
class DataSet:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.paths, self.labels = DataSet.generate_paths(self.data_dir)
        self.paths, self.labels = shuffle_image_label(self.paths, self.labels)
    @staticmethod
    def extract_ROI_image(mhd_path, tumor_path, liver_path, image_size):

        mhd_image = read_mhd_image(mhd_path, rejust=True)
        # show_image(mhd_image)
        tumor_mask_image = read_mhd_image(tumor_path)
        liver_mask_image = read_mhd_image(liver_path)
        mhd_image = compress22dim(mhd_image)
        tumor_mask_image = compress22dim(tumor_mask_image)
        liver_mask_image = compress22dim(liver_mask_image)
        # 注意记得填充图像
        tumor_bounding_box = get_boundingbox(tumor_mask_image)
        liver_average = cal_liver_average(mhd_image, liver_mask_image)
        mhd_image_copy = copy(mhd_image)
        mhd_image_copy[tumor_mask_image != 1] = 0
        roi_image = mhd_image_copy[
                    tumor_bounding_box[0]: tumor_bounding_box[1],
                    tumor_bounding_box[2]: tumor_bounding_box[3]
                    ]
        roi_image = resize_image(roi_image, size=image_size)
        roi_image = np.asarray(roi_image, np.float32)
        roi_image = roi_image / liver_average
        return roi_image
    def generate_next_images_batch(self, batch_size, image_size):
        '''
        返回一个batch 的image
        :param batch_size: batch 的大小
        :param image_size: 图像的大小
        :return:
        '''

        batch_paths, batch_indexs = self.generate_next_paths_batch(batch_size)
        res_images = []
        res_labels = []
        for first_index, pcid_paths in enumerate(batch_paths):
            # 同一个pcid 的path
            phase_images = np.zeros([image_size, image_size, 3], dtype=np.float32)
            for index, phase_paths in enumerate(pcid_paths):
                # 同一个phase的path
                roi_image = DataSet.extract_ROI_image(phase_paths[0], phase_paths[1], phase_paths[2], image_size)
                phase_images[:, :, index] = roi_image
            res_images.append(phase_images)
            res_labels.append(self.labels[batch_indexs[first_index]])
        return np.array(res_images), np.array(res_labels)

    def generate_next_paths_batch(self, batch_size):
        '''
        返回一个batch 的paths
        :param batch_size: batch 的大小
        :return:
        '''
        res_paths = []
        res_indexs = []
        for i in range(batch_size):
            cur_index = np.random.randint(0, len(self.paths), 1)[0]
            res_paths.append(self.paths[cur_index])
            res_indexs.append(cur_index)
        return res_paths, res_indexs
    @staticmethod
    def generate_paths(dir_name):
        '''
        根据目录生成图像的路径
        :param dir_name: 数据集的目录
        :return: paths array
        '''
        subnames = os.listdir(dir_name)
        res_paths = []
        labels = []
        for subname in subnames:
            cur_dir = os.path.join(dir_name, subname)
            diffPhasePaths = []
            for phasename in ['NC', 'ART', 'PV']:
                cur_paths = []
                mhd_path = glob(os.path.join(cur_dir, phasename + '_Image*.mhd'))[0]
                mask_paths = glob(os.path.join(cur_dir, phasename + '_Mask.mhd'))
                if len(mask_paths) == 0:
                    mask_paths = glob(os.path.join(cur_dir, phasename + '_Mask_*.mhd'))
                    for path in mask_paths:
                        if path.endswith('Expand.mhd'):
                            continue
                        else:
                            mask_path = path
                            break
                else:
                    mask_path = mask_paths[0]
                # print mask_path
                liver_path = glob(os.path.join(cur_dir, phasename + '*Liver*.mhd'))[0]
                cur_paths.append(mhd_path)
                cur_paths.append(mask_path)
                cur_paths.append(liver_path)
                diffPhasePaths.append(cur_paths)
            res_paths.append(diffPhasePaths)
            labels.append(int(cur_dir[-1]))
        return res_paths, labels

if __name__ == '__main__':
    dataset = DataSet('/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/train')
    image_batch, label_batch = dataset.generate_next_images_batch(20, 256)
    print label_batch
    for index, image in enumerate(image_batch):
        print 'label is ', label_batch[index]
        for index in range(3):
            mean_valus = np.sum(image[:, :, index])
            point_number = np.sum(image[:, :, index] != 0)
            print mean_valus / point_number
        # show_image(np.asarray(image + 1, np.uint8) * 100)
    # print np.shape(image_batch)
    # DataSet.extract_ROI_image(
    #
    # )