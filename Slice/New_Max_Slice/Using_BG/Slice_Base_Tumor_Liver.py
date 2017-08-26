# -*- coding: utf-8 -*-
# 主要是重写ｅｘｔｒａｃｔ_ｒｏｉ方法
# 该方法将提取病灶部分作为我们的ＲＯＩ
from Slice_Base import Slice_Base
from Config import Config
import numpy as np
from PIL import Image


class Liver_Tumor_Operations:
    # 两者相加
    @staticmethod
    def liver_add_tumor(liver_image, tumor_image):
        return liver_image + tumor_image

    # 病灶除以肝脏的平均密度
    @staticmethod
    def tumor_div_average_liver(liver_image, tumor_image):
        sumed_index = np.logical_and(liver_image != 0, tumor_image == 0)
        pixel_sum = np.sum(liver_image[np.where(sumed_index == True)])
        pixel_num = np.sum(sumed_index)
        average_pixel_value = (pixel_sum * 1.0) / (pixel_num * 1.0)
        print average_pixel_value
        return (tumor_image * 1.0) / average_pixel_value

    # x = avg(tumor) - avg(liver)
    # new_lesion = x * new_lesion
    @staticmethod
    def tumor_linear_enhancement(liver_image, tumor_image):
        # show_image(tumor_image, 'before')
        sumed_index = np.logical_and(liver_image != 0, tumor_image == 0)
        pixel_sum = np.sum(liver_image[np.where(sumed_index == True)])
        pixel_num = np.sum(sumed_index)
        average_liver_pixel_value = (pixel_sum * 1.0) / (pixel_num * 1.0)
        pixel_sum = np.sum(tumor_image)
        pixel_num = np.sum(tumor_image != 0)
        average_tumor_pixel_value = (pixel_sum * 1.0) / (pixel_num * 1.0)
        x = average_tumor_pixel_value / average_liver_pixel_value
        # print 'linear rate is ', x
        new_tumor = tumor_image * x
        # show_image(new_tumor, 'after')
        return new_tumor


class Slice_Base_Tumor_Liver(Slice_Base):
    def __init__(self, path, operation):
        # self.extract_roi_function = self.extract_roi_resize
        self.extract_roi_function = self.extract_roi_diff_size
        self.extract_roi_bg_function = self.extract_roi_bg_diff_resize
        self.operation = operation
        Slice_Base.__init__(self, path)

    def extract_roi(self):
        self.extract_roi_function()

    def extract_roi_bg(self):
        self.extract_roi_bg_function()

    def extract_roi_bg_resize(self, new_size=[60, 60]):
        self.rois_bg = []
        for diff_phase_lession in self.lesion_bg:
            new_diff_phase_lession = []
            for single_phase_lession in diff_phase_lession:
                [xs, ys] = np.where(single_phase_lession != 0)
                min_xs = np.min(xs)
                max_xs = np.max(xs)
                min_ys = np.min(ys)
                max_ys = np.max(ys)
                single_phase_lession = single_phase_lession[min_xs:max_xs+1, min_ys:max_ys+1]
                image = Image.fromarray(single_phase_lession)
                image = image.resize(new_size)
                new_diff_phase_lession.append(
                    np.array(image)
                )
            self.rois_bg.append(new_diff_phase_lession)
        self.rois_bg = np.array(self.rois_bg)

    def extract_roi_bg_diff_resize(self, new_sizes=[[60, 60], [35, 35], [120, 120]]):
        self.rois_bg = []
        for index, diff_phase_bg in enumerate(self.lesion_bg):
            new_diff_size_phase_lesion = []
            for new_size in new_sizes:
                new_diff_phase_bg = []
                for index_phase, single_phase_bg in enumerate(diff_phase_bg):
                    [xs, ys] = np.where(single_phase_bg != 0)
                    min_xs = np.min(xs)
                    max_xs = np.max(xs)
                    min_ys = np.min(ys)
                    max_ys = np.max(ys)
                    single_phase_bg = single_phase_bg[min_xs:max_xs + 1, min_ys:max_ys + 1]
                    image = Image.fromarray(single_phase_bg)
                    # print new_size
                    image = image.resize(new_size)
                    new_diff_phase_bg.append(
                        np.array(image)
                    )
                new_diff_size_phase_lesion.append(new_diff_phase_bg)
            self.rois_bg.append(new_diff_size_phase_lesion)
        self.rois_bg = np.array(self.rois_bg)
        print 'finish extract_roi_bg_diff_resize, rois length is ', len(self.rois_bg)

    def extract_roi_resize(self, new_size=[45, 45]):
        self.rois = []
        for index, diff_phase_liver in enumerate(self.livers):
            new_diff_phase_lession = []
            print 'label is ', self.labels[index]
            for index_phase, single_phase_liver in enumerate(diff_phase_liver):
                single_phase_lession = self.operation(single_phase_liver, self.lesions[index][index_phase])
                [xs, ys] = np.where(single_phase_lession != 0)
                min_xs = np.min(xs)
                max_xs = np.max(xs)
                min_ys = np.min(ys)
                max_ys = np.max(ys)
                single_phase_lession = single_phase_lession[min_xs:max_xs+1, min_ys:max_ys+1]
                image = Image.fromarray(np.asarray(single_phase_lession, np.float32))
                image = image.resize(new_size)
                new_diff_phase_lession.append(
                    np.array(image)
                )
            self.rois.append(new_diff_phase_lession)
        self.rois = np.array(self.rois)

    def extract_roi_diff_size(self, new_sizes=[[45, 45], [20, 20], [100, 100]]):
        self.rois = []
        for index, diff_phase_liver in enumerate(self.livers):
            new_diff_size_phase_lesion = []
            for new_size in new_sizes:
                new_diff_phase_lession = []
                for index_phase, single_phase_liver in enumerate(diff_phase_liver):
                    single_phase_lesion = self.operation(single_phase_liver, self.lesions[index][index_phase])
                    [xs, ys] = np.where(single_phase_lesion != 0)
                    min_xs = np.min(xs)
                    max_xs = np.max(xs)
                    min_ys = np.min(ys)
                    max_ys = np.max(ys)
                    single_phase_lesion = single_phase_lesion[min_xs:max_xs + 1, min_ys:max_ys + 1]
                    image = Image.fromarray(single_phase_lesion)
                    # print new_size
                    image = image.resize(new_size)
                    new_diff_phase_lession.append(
                        np.array(image)
                    )
                new_diff_size_phase_lesion.append(new_diff_phase_lession)
            self.rois.append(new_diff_size_phase_lesion)
        self.rois = np.array(self.rois)
        print 'finish extract_roi_diff_size, rois length is ', len(self.rois)


class SliceBaseTumorLiverDataset():
    def __init__(self, config, operation):
        self.train_dataset = Slice_Base_Tumor_Liver(config.TRAIN_IMAGE_DIR, operation)
        self.val_dataset = Slice_Base_Tumor_Liver(config.VALIDATION_IMAGE_DIR, operation)

    def next_train_batch(self, batch_size, distribution=None):
        return self.train_dataset.get_next_batch(batch_size, distribution)

    def next_val_batch(self, batch_size, distribution=None):
        return self.val_dataset.get_next_batch(batch_size, distribution)

if __name__ == '__main__':
    dataset = SliceBaseTumorLiverDataset(Config, Liver_Tumor_Operations.tumor_linear_enhancement)
    images, labels, bgs = dataset.next_train_batch(20)
    print labels
    images, labels, bgs = dataset.next_val_batch(20)
    print labels