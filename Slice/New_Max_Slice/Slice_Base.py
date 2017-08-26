# -*- coding: utf-8 -*-
from PIL import Image
from Config import Config
import os
import numpy as np
from Tools import shuffle_image_label

class Slice_Base:
    def __init__(self, config):
        self.config = config
        self.rois = []
        self.train_images = []
        self.train_labels = []
        self.validation_images = []
        self.validation_labels = []
        self.epoch_num = 0
        self.start_index = 0
        self.livers, self.lesions, self.labels = Slice_Base.load_lesions_labels(self.config.IMAGE_DIR)
        self.extract_roi()  # 提取ＲＯＩ,这个是由子类完成的
        self.split_train_and_validation()   # 拆分为训练集合和验证集合


    @staticmethod
    def load_lesions_labels(data_path):
        def open_image_by_path(path):
            if np.sum(Image.open(path)) == 0:
                print 'Error, ', path
            return np.array(
                Image.open(path)
            )
        lesions_files = os.listdir(data_path)
        lesions = []
        labels = []
        livers = []
        for lesion in lesions_files:
            label = int(lesion[-1])
            labels.append(label)
            liver_art_path = os.path.join(data_path, lesion, 'liver_art.jpg')
            liver_nc_path = os.path.join(data_path, lesion, 'liver_nc.jpg')
            liver_pv_path = os.path.join(data_path, lesion, 'liver_pv.jpg')
            tumor_nc_path = os.path.join(data_path, lesion, 'tumor_nc.jpg')
            tumor_art_path = os.path.join(data_path, lesion, 'tumor_art.jpg')
            tumor_pv_path = os.path.join(data_path, lesion, 'tumor_pv.jpg')
            diff_phase_liver = [
                open_image_by_path(liver_nc_path),
                open_image_by_path(liver_art_path),
                open_image_by_path(liver_pv_path)
            ]
            diff_phase_tumor = [
                open_image_by_path(tumor_nc_path),
                open_image_by_path(tumor_art_path),
                open_image_by_path(tumor_pv_path)
            ]
            lesions.append(diff_phase_tumor)
            livers.append(diff_phase_liver)
        return np.array(livers), \
               np.array(lesions), \
               np.array(labels)

    def extract_roi(self):
        print 'excute Slice Base extract roi'
        print 'liver shape is ', np.shape(self.livers)
        self.rois = []

    # 按照一定分布将数据集拆分为训练集和测试集
    def split_train_and_validation(self):
        validation_lesions = []
        validation_labels = []
        train_lesions = []
        train_labels = []
        for index in range(len(self.config.LESION_TYPE)):
            # 先挑出这个类型的所有病灶
            lesions = self.rois[np.where(self.labels == index)]
            labels = self.labels[np.where(self.labels == index)]
            random_index = range(len(lesions))
            # np.random.shuffle(random_index)
            lesions = lesions[random_index]
            labels = labels[random_index]
            validation_num = self.config.VALIDATION_DISTRIBUTION[index]
            validation_lesions.extend(lesions[:validation_num])
            train_lesions.extend(lesions[validation_num:])
            validation_labels.extend(labels[: validation_num])
            train_labels.extend(labels[validation_num:])
        print 'validation shape is ', np.shape(validation_lesions)
        print 'train shape is ', np.shape(train_lesions)
        self.validation_images, self.validation_labels = shuffle_image_label(validation_lesions, validation_labels)
        self.train_images, self.train_labels = shuffle_image_label(train_lesions, train_labels)
        print 'validation label is \n', self.validation_labels
        print 'train_label is \n', self.train_labels

    # 获取验证集数据
    def get_validation_images_labels(self):
        return self.validation_images, self.validation_labels

    # 获取next batch data
    # distribution 是指是否按照一定的比例来去batch
    def get_next_batch(self, batch_size, distribution=None):
        end_index = self.start_index + batch_size
        images = []
        labels = []
        if distribution is None:
            if end_index >= len(self.train_images):
                images.extend(
                    self.train_images[self.start_index: len(self.train_images)]
                )
                images.extend(
                    self.train_images[:end_index - len(self.train_images)]
                )
                labels.extend(
                    self.train_labels[self.start_index: len(self.train_images)]
                )
                labels.extend(
                    self.train_labels[:end_index - len(self.train_images)]
                )
                self.start_index = end_index - len(self.train_images)
                self.epoch_num += 1
                # print self.epoch_num
            else:
                images.extend(
                    self.train_images[self.start_index: end_index]
                )
                labels.extend(
                    self.train_labels[self.start_index: end_index]
                )
                self.start_index = end_index
        else:
            for index, num in enumerate(distribution):
                target_indexs = np.where(self.train_labels == index)[0]
                np.random.shuffle(target_indexs)
                images.extend(self.train_images[target_indexs[:num]])
                labels.extend(self.train_labels[target_indexs[:num]])
            images, labels = shuffle_image_label(images, labels)
        return images, labels