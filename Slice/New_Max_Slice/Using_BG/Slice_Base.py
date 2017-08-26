# -*- coding: utf-8 -*-
from PIL import Image
from Config import Config
import os
import numpy as np
from Tools import shuffle_image_label, get_shuffle_index


class Slice_Base:
    def __init__(self, path):
        self.rois = []
        self.epoch_num = 0
        self.start_index = 0
        self.rois_bg = []
        self.livers, self.lesions, self.labels, self.lesion_bg = Slice_Base.load_lesions_labels(path)
        self.extract_roi()  # 提取ＲＯＩ,这个是由子类完成的
        self.extract_roi_bg()  # 提取背景的ＲＯＩ,这个是由子类完成的
        self.shuffle_roi_label()
        # self.split_train_and_validation()   # 拆分为训练集合和验证集合

    def shuffle_roi_label(self):
        self.rois = np.array(self.rois)
        self.rois_bg = np.array(self.rois_bg)
        self.labels = np.array(self.labels)
        random_index = get_shuffle_index(len(self.rois_bg))
        self.rois_bg = self.rois_bg[random_index]
        self.rois = self.rois[random_index]
        self.labels = self.labels[random_index]

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
        lesions_bg = []
        for lesion in lesions_files:
            label = int(lesion[-1])
            labels.append(label)
            liver_art_path = os.path.join(data_path, lesion, 'liver_art.jpg')
            liver_nc_path = os.path.join(data_path, lesion, 'liver_nc.jpg')
            liver_pv_path = os.path.join(data_path, lesion, 'liver_pv.jpg')
            tumor_nc_path = os.path.join(data_path, lesion, 'tumor_nc.jpg')
            tumor_art_path = os.path.join(data_path, lesion, 'tumor_art.jpg')
            tumor_pv_path = os.path.join(data_path, lesion, 'tumor_pv.jpg')
            lesions_art_bg = os.path.join(data_path, lesion, 'tumor_art_bg.jpg')
            lesions_nc_bg = os.path.join(data_path, lesion, 'tumor_nc_bg.jpg')
            lesions_pv_bg = os.path.join(data_path, lesion, 'tumor_pv_bg.jpg')
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
            diff_phase_tumor_bg = [
                open_image_by_path(lesions_nc_bg),
                open_image_by_path(lesions_art_bg),
                open_image_by_path(lesions_pv_bg)
            ]
            lesions.append(diff_phase_tumor)
            livers.append(diff_phase_liver)
            lesions_bg.append(diff_phase_tumor_bg)
        return np.array(livers), \
               np.array(lesions), \
               np.array(labels), \
               np.array(lesions_bg)

    def extract_roi(self):
        print 'excute Slice Base extract roi'
        print 'liver shape is ', np.shape(self.livers)
        self.rois = []

    def extract_roi_bg(self):
        print 'Slice Base'
        self.rois_bg = []

    '''
    # 按照一定分布将数据集拆分为训练集和测试集
    def split_train_and_validation(self):
        validation_lesions = []
        validation_labels = []
        validation_bg = []
        train_lesions = []
        train_labels = []
        train_bg = []
        for index in range(len(self.config.LESION_TYPE)):
            # 先挑出这个类型的所有病灶
            lesions = self.rois[np.where(self.labels == index)]
            labels = self.labels[np.where(self.labels == index)]
            bgs = self.rois_bg[np.where(self.labels == index)]
            random_index = range(len(lesions))
            # np.random.shuffle(random_index)
            lesions = lesions[random_index]
            labels = labels[random_index]
            bgs = bgs[random_index]
            validation_num = self.config.VALIDATION_DISTRIBUTION[index]
            validation_lesions.extend(lesions[:validation_num])
            train_lesions.extend(lesions[validation_num:])
            validation_labels.extend(labels[: validation_num])
            train_labels.extend(labels[validation_num:])
            train_bg.extend(bgs[validation_num:])
            validation_bg.extend(bgs[: validation_num])
        print 'validation shape is ', np.shape(validation_lesions)
        print 'train shape is ', np.shape(train_lesions)
        shuffle_index = get_shuffle_index(len(validation_labels))
        validation_lesions = np.array(validation_lesions)
        validation_labels = np.array(validation_labels)
        validation_bg = np.array(validation_bg)
        train_lesions = np.array(train_lesions)
        train_labels = np.array(train_labels)
        train_bg = np.array(train_bg)
        self.validation_images = validation_lesions[shuffle_index]
        self.validation_labels = validation_labels[shuffle_index]
        self.validation_bg = validation_bg[shuffle_index]
        shuffle_index = get_shuffle_index(len(train_lesions))
        self.train_images = train_lesions[shuffle_index]
        self.train_labels = train_labels[shuffle_index]
        self.train_bg = train_bg[shuffle_index]
        print 'validation label is \n', self.validation_labels
        print 'train_label is \n', self.train_labels

    '''

    # 获取验证集数据
    def get_validation_images_labels(self):
        return self.validation_images, self.validation_labels, self.validation_bg

    # 获取next batch data
    # distribution 是指是否按照一定的比例来去batch
    def get_next_batch(self, batch_size, distribution=None):
        end_index = self.start_index + batch_size
        images = []
        labels = []
        bgs = []
        if distribution is None:
            if end_index >= len(self.rois):
                images.extend(
                    self.rois[self.start_index: len(self.rois)]
                )
                images.extend(
                    self.rois[:end_index - len(self.rois)]
                )
                labels.extend(
                    self.labels[self.start_index: len(self.rois)]
                )
                labels.extend(
                    self.labels[:end_index - len(self.rois)]
                )
                bgs.extend(
                    self.rois_bg[self.start_index: len(self.rois)]
                )
                bgs.extend(
                    self.rois_bg[:end_index - len(self.rois)]
                )
                self.start_index = end_index - len(self.rois)
                self.epoch_num += 1
                # print self.epoch_num
            else:
                images.extend(
                    self.rois[self.start_index: end_index]
                )
                labels.extend(
                    self.labels[self.start_index: end_index]
                )
                bgs.extend(self.rois_bg[self.start_index: end_index])
                self.start_index = end_index
        else:
            for index, num in enumerate(distribution):
                target_indexs = np.where(self.labels == index)[0]
                np.random.shuffle(target_indexs)
                images.extend(self.rois[target_indexs[:num]])
                labels.extend(self.labels[target_indexs[:num]])
                bgs.extend(self.rois_bg[target_indexs[:num]])
        images = np.array(images)
        labels = np.array(labels)
        bgs = np.array(bgs)
        shuffle_index = get_shuffle_index(len(images))
        images = images[shuffle_index]
        labels = labels[shuffle_index]
        bgs = bgs[shuffle_index]
        return images, labels, bgs
