# -*- coding: utf-8 -*-
from Slice.MaxSlice_Liver.MaxSlice_Liver_Base import MaxSlice_Liver_Base
import numpy as np
from PIL import Image
from Config import Config
from Tools import save_image, get_distribution_label, show_image
import os


class MaxSlice_Liver_Resize(MaxSlice_Liver_Base):
    def __init__(self, config):
        self.config = config
        MaxSlice_Liver_Base.__init__(self, config)
        self.images, self.masks, self.labels = MaxSlice_Liver_Base.load_image_mask_label(config)
        print 'load images shape is ', np.shape(self.images)
        self.roi_images = MaxSlice_Liver_Resize.resize_images(self.images, self.masks, config.MaxSlice_Resize['RESIZE'])
        print np.shape(self.roi_images)
        self.save_ROI_image(self.config.MaxSlice_Resize['IMAGE_SAVE_PATH'])
        self.start_index = 0
        self.epoch_num = 0
        self.roi_images = np.array(self.roi_images)
        self.labels = np.array(self.labels)
        self.shuffle_ROI()
        self.split_train_and_validation()

    @staticmethod
    def resize_images(images, masks, new_size):
        roi_images = []
        print np.shape(images)
        print np.shape(masks)
        for index, phase_images in enumerate(images):
            cur_roi_images = []
            for phase_index, image in enumerate(phase_images):
                # if index == 0:
                #     show_image(image * 120)
                mask_image = masks[index][phase_index]
                [ys, xs] = np.where(mask_image != 0)
                miny = np.min(ys)
                maxy = np.max(ys)
                minx = np.min(xs)
                maxx = np.max(xs)
                ROI = image[miny - 2:maxy + 2, minx - 2:maxx + 2]
                ROI_Image = Image.fromarray(np.asarray(ROI, np.uint8))
                ROI_Image = ROI_Image.resize(new_size)
                cur_roi_images.append(np.array(ROI_Image))
            roi_images.append(cur_roi_images)
        return roi_images

    @staticmethod
    def test_unit():
        # MaxSlice_Resize(Config)
        dataset = MaxSlice_Liver_Resize(Config)
        batch_size = 20
        for i in range(100):
            images, labels = dataset.get_next_batch(batch_size, [10, 10, 10, 10, 10])
            print np.shape(images), np.shape(labels)
            print get_distribution_label(labels)
if __name__ == '__main__':
    MaxSlice_Liver_Resize.test_unit()
