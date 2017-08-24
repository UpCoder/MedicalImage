# -*- coding: utf-8 -*-
import numpy as np
import os
from Config import Config
import scipy.io as scio
import gc
from ExcelData import ExcelData
import glob
from Tools import save_image, get_lesion_type_by_srrid, get_diff_phases_images, get_total_masks, save_image_with_mask, shuffle_image_label


class MaxSlice_Liver_Base:
    def __init__(self, config):
        self.lesions_data = ExcelData().lesions_by_srrid
        if config.MaxSlice_Base['splited']['statue']:
            for range_index, cur_range in enumerate(config.MaxSlice_Base['splited']['ranges']):
                if not os.path.exists(Config.MaxSlice_Base['splited']['save_paths'][range_index][0]):

                    self.images, self.labels = MaxSlice_Liver_Base.load_images_labels(config, self.lesions_data, cur_range)
                    np.save(
                        Config.MaxSlice_Base['splited']['save_paths'][range_index][0],
                        self.images
                    )
                    np.save(
                        Config.MaxSlice_Base['splited']['save_paths'][range_index][2],
                        self.labels
                    )
                    del self.images, self.labels
                    gc.collect()
                if not os.path.exists( Config.MaxSlice_Base['splited']['save_paths'][range_index][1]):
                    self.masks = MaxSlice_Liver_Base.load_masks(config, self.lesions_data, cur_range)
                    np.save(
                        Config.MaxSlice_Base['splited']['save_paths'][range_index][1],
                        self.masks
                    )
                    del self.masks
                    gc.collect()
        else:
            if os.path.exists(Config.MaxSliceDataPATH[0]):
                self.images = np.load(
                    Config.MaxSliceDataPATH[0]
                )
                self.masks = np.load(
                    Config.MaxSliceDataPATH[1]
                )
                self.labels = np.load(
                    Config.MaxSliceDataPATH[2]
                )
                return
            self.images, self.labels = MaxSlice_Liver_Base.load_images_labels(config, self.lesions_data, range(0, 200))
            np.save(
                Config.MaxSliceDataPATH[0],
                self.images
            )
            np.save(
                Config.MaxSliceDataPATH[2],
                self.labels
            )
            del self.images, self.labels
            gc.collect()
            self.masks = MaxSlice_Liver_Base.load_masks(config, self.lesions_data, range(0, 200))
            np.save(
                Config.MaxSliceDataPATH[1],
                self.masks
            )

    # 将ＲＯＩ保存成图片
    def save_ROI_image(self, path):
        for index, roi_images_phase in enumerate(self.roi_images):
            for phase_index, roi_image in enumerate(roi_images_phase):
                save_image(roi_image, os.path.join(path, str(index) + '_' + str(phase_index) + '.jpg'))

    # 按照一定分布将数据集拆分为训练集和测试集
    def split_train_and_validation(self):
        validation_lesions = []
        validation_labels = []
        train_lesions = []
        train_labels = []
        for index in range(len(Config.LESION_TYPE)):
            # 先挑出这个类型的所有病灶
            lesions = self.roi_images[np.where(self.labels == index)]
            labels = self.labels[np.where(self.labels == index)]
            random_index = range(len(lesions))
            # np.random.shuffle(random_index)
            lesions = lesions[random_index]
            labels = labels[random_index]
            validation_num = Config.MaxSlice_Base['VALIDATION_DISTRIBUTION'][index]
            validation_lesions.extend(lesions[:validation_num])
            train_lesions.extend(lesions[validation_num:])
            validation_labels.extend(labels[: validation_num])
            train_labels.extend(labels[validation_num: ])
        print 'validation shape is ', np.shape(validation_lesions)
        print 'train shape is ', np.shape(train_lesions)
        self.validation_images, self.validation_labels = shuffle_image_label(validation_lesions, validation_labels)
        self.train_images, self.train_labels = shuffle_image_label(train_lesions, train_labels)
        print 'validation label is \n', self.validation_labels
        print 'train_label is \n', self.train_labels

    # 将数据打乱
    def shuffle_ROI(self):
        print 'Before Shuffle', self.labels
        print 'roiimages len is ', len(self.roi_images)
        random_index = range(len(self.roi_images))
        np.random.shuffle(random_index)
        self.roi_images = self.roi_images[random_index]
        self.labels = self.labels[random_index]
        print 'After Shuffle', self.labels

    # 获取验证集数据
    def get_validation_images_labels(self):
        return self.validation_images, self.validation_labels

    # 获取next batch data
    # distribution 是指是否按照一定的比例来去batch
    def get_next_batch(self, batch_size, distribution = None):
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

    @staticmethod
    def load_image_mask_label(config):
        mask_files = glob.glob(config.MaxSlice_Liver_Base['BASE_DATA_PATH'] + '/MaxSlice_Mask*.npy')
        image_files = glob.glob(config.MaxSlice_Liver_Base['BASE_DATA_PATH'] + '/MaxSlice_Image*.npy')
        label_files = glob.glob(config.MaxSlice_Liver_Base['BASE_DATA_PATH'] + '/MaxSlice_Label*.npy')
        masks = []
        for mask_file in mask_files:
            masks.extend(
                np.load(mask_file)
            )
        images = []
        for image_file in image_files:
            images.extend(
                np.load(image_file)
            )
        labels = []
        for label_file in label_files:
            labels.extend(
                np.load(label_file)
            )
        print labels
        return images, masks, labels

    @staticmethod
    def cout_mask_size(config):
        if config.MaxSlice_Base['splited']['statue']:
            masks = []
            avg_y = 0.0
            avg_x = 0.0
            min_y = 9999999  # y方向上最小的差距
            max_y = 0.0  # y方向上最大的差距
            min_x = 9999999  # x方向上最小的差距
            max_x = 0.0  # x方向上最大的差距
            masks_file = glob.glob(config.MaxSlice_Liver_Base['BASE_DATA_PATH'] + '/MaxSlice_Mask*.npy')
            for mask_file in masks_file:
                masks.extend(
                    np.load(
                        mask_file
                    )
                )
            for index, phase_mask in enumerate(masks):
                print 'index is ', index
                for mask in phase_mask:
                    [ys, xs] = np.where(mask != 0)
                    if len(ys) == 0:
                        print 'Error'
                        continue
                    miny = np.min(ys)
                    maxy = np.max(ys)
                    min_y = min(min_y, (maxy - miny))
                    max_y = max(max_y, (maxy - miny))
                    minx = np.min(xs)
                    maxx = np.max(xs)
                    min_x = min(min_x, (maxx - minx))
                    max_x = max(max_x, (maxx - minx))
                    avg_y += (maxy - miny)
                    avg_x += (maxx - minx)
                    print '(%d, %d)' % (maxy - miny, maxx - minx)
            print '(%f, %f)' % (avg_y / (len(masks) * 3), avg_x / (len(masks) * 3))
            print 'max is (%d, %d)' % (max_y, max_x)
            print 'min is (%d, %d)' % (min_y, min_x)
        else:
            mask_file_path = config.MaxSliceDataPATH[1]
            masks = np.load(mask_file_path)
            avg_y = 0.0
            avg_x = 0.0
            min_y = 9999999     # y方向上最小的差距
            max_y = 0.0     # y方向上最大的差距
            min_x = 9999999     # x方向上最小的差距
            max_x = 0.0     # x方向上最大的差距
            for index, phase_mask in enumerate(masks):
                print 'index is ', index
                for mask in phase_mask:
                    [ys, xs] = np.where(mask != 0)
                    if len(ys) == 0:
                        print 'Error'
                        continue
                    miny = np.min(ys)
                    maxy = np.max(ys)
                    min_y = min(min_y, (maxy - miny))
                    max_y = max(max_y, (maxy - miny))
                    minx = np.min(xs)
                    maxx = np.max(xs)
                    min_x = min(min_x, (maxx - minx))
                    max_x = max(max_x, (maxx - minx))
                    avg_y += (maxy-miny)
                    avg_x += (maxx-minx)
                    print '(%d, %d)' % (maxy-miny, maxx-minx)
            print '(%f, %f)' % (avg_y/(len(masks)*3), avg_x/(len(masks)*3))
            print 'max is (%d, %d)' % (max_y, max_x)
            print 'min is (%d, %d)' % (min_y, min_x)

    @staticmethod
    def load_images_labels(config, lesions_data, cur_range):
        image_slices = []
        mask_slices = []
        labels = []
        for key in lesions_data.keys():
            if key not in cur_range:
                continue
            srrid = key
            str_srrid = '%03d' % srrid
            lesion_type = get_lesion_type_by_srrid(srrid)
            images_path = glob.glob(os.path.join(config.DATASET_PATH, lesion_type, str_srrid + '-*'))[0]
            images = get_diff_phases_images(images_path)
            masks = get_total_masks(images_path)
            tumors_mask = masks['TumorMask']
            print lesions_data[key]
            for index, lesion in enumerate(lesions_data[key]):
                nap_index = lesion[1:]
                print nap_index
                cur_image_slice = []
                if key != 45:
                    nc_image_slice = images['NC'][nap_index[0] - 1, :, :]
                    art_image_slice = images['ART'][nap_index[1] - 1, :, :]
                else:
                    nc_image_slice = images['NC'][np.shape(images['NC'])[0] - nap_index[0], :, :]
                    art_image_slice = images['ART'][np.shape(images['ART'])[0] - nap_index[1], :, :]
                if key in [45, 177]:
                    pv_image_slice = images['PV'][np.shape(images['PV'])[0] - nap_index[2], :, :]
                else:
                    pv_image_slice = images['PV'][nap_index[2] - 1, :, :]
                nc_mask_slice = tumors_mask[index]['NC'][np.shape(images['NC'])[0] - nap_index[0], :, :]
                art_mask_slice = tumors_mask[index]['ART'][np.shape(images['ART'])[0] - nap_index[1], :, :]
                pv_mask_slice = tumors_mask[index]['PV'][np.shape(images['PV'])[0] - nap_index[2], :, :]
                cur_image_slice.append(nc_image_slice)
                cur_image_slice.append(art_image_slice)
                cur_image_slice.append(pv_image_slice)
                image_slices.append(cur_image_slice)
                labels.append(Config.LABEL_NUMBER_MAPPING[lesion_type])
                save_image_with_mask(nc_image_slice, nc_mask_slice, os.path.join(config.IMAGE_SAVE_PATH, str(srrid) + '_' + str(index + 1) + '_nc_image_mask.jpg'))
                save_image_with_mask(art_image_slice, art_mask_slice, os.path.join(config.IMAGE_SAVE_PATH, str(srrid) + '_' + str(index + 1) + '_art_image_mask.jpg'))
                save_image_with_mask(pv_image_slice, pv_mask_slice, os.path.join(config.IMAGE_SAVE_PATH, str(srrid) + '_' + str(index + 1) + '_pv_image_mask.jpg'))
                # print key, nap_index
        return image_slices, labels

    @staticmethod
    def load_masks(config, lesions_data, cur_range):
        image_slices = []
        mask_slices = []
        labels = []
        for key in lesions_data.keys():
            if key not in cur_range:
                continue
            srrid = key
            str_srrid = '%03d' % srrid
            lesion_type = get_lesion_type_by_srrid(srrid)
            images_path = glob.glob(os.path.join(config.DATASET_PATH, lesion_type, str_srrid + '-*'))[0]
            images = get_diff_phases_images(images_path)
            masks = get_total_masks(images_path)
            tumors_mask = masks['TumorMask']
            print lesions_data[key]
            for index, lesion in enumerate(lesions_data[key]):
                nap_index = lesion[1:]
                print nap_index
                # print np.shape(images['NC']), np.shape(images['ART']), np.shape(images['PV'])[0]
                cur_mask_slice = []
                nc_mask_slice = tumors_mask[index]['NC'][np.shape(images['NC'])[0] - nap_index[0], :, :]
                art_mask_slice = tumors_mask[index]['ART'][np.shape(images['ART'])[0] - nap_index[1], :, :]
                pv_mask_slice = tumors_mask[index]['PV'][np.shape(images['PV'])[0] - nap_index[2], :, :]
                cur_mask_slice.append(nc_mask_slice)
                cur_mask_slice.append(art_mask_slice)
                cur_mask_slice.append(pv_mask_slice)
                mask_slices.append(cur_mask_slice)
                # print key, nap_index
        return mask_slices

    @staticmethod
    def test_unit():
        MaxSlice_Liver_Base.cout_mask_size(Config)
        # dataset = MaxSlice_Base(Config)
        # print len(dataset.images)
if __name__ == '__main__':
    MaxSlice_Liver_Base.test_unit()