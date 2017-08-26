# -*- coding: utf-8 -*-
# 主要是重写ｅｘｔｒａｃｔ_ｒｏｉ方法
# 该方法将提取病灶部分作为我们的ＲＯＩ
from Slice_Base import Slice_Base
from Config import Config
import numpy as np
from PIL import Image


class Slice_Base_Tumor(Slice_Base):
    def __init__(self, path):
        # self.extract_roi_function = self.extract_roi_resize
        self.extract_roi_function = self.extract_roi_resize
        self.extract_roi_bg_function = self.extract_roi_bg_resize
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

    def extract_roi_resize(self, new_size=[45, 45]):
        self.rois = []
        for diff_phase_lession in self.lesions:
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
            self.rois.append(new_diff_phase_lession)
        self.rois = np.array(self.rois)

    def extract_roi_diff_size(self, new_sizes=[[45, 45], [20, 20], [100, 100]]):
        self.rois = []
        for diff_phase_lesion in self.lesions:
            new_diff_size_phase_lesion = []
            for new_size in new_sizes:
                new_diff_phase_lession = []
                for single_phase_lesion in diff_phase_lesion:
                    [xs, ys] = np.where(single_phase_lesion != 0)
                    min_xs = np.min(xs)
                    max_xs = np.max(xs)
                    min_ys = np.min(ys)
                    max_ys = np.max(ys)
                    single_phase_lesion = single_phase_lesion[min_xs:max_xs + 1, min_ys:max_ys + 1]
                    image = Image.fromarray(single_phase_lesion)
                    image = image.resize(new_size)
                    new_diff_phase_lession.append(
                        np.array(image)
                    )
                new_diff_size_phase_lesion.append(new_diff_phase_lession)
            self.rois.append(new_diff_size_phase_lesion)
        self.rois = np.array(self.rois)


class SliceBaseTumorDataset():
    def __init__(self, config):
        self.train_dataset = Slice_Base_Tumor(config.TRAIN_IMAGE_DIR)
        self.val_dataset = Slice_Base_Tumor(config.VALIDATION_IMAGE_DIR)

    def next_train_batch(self, batch_size, distribution=None):
        return self.train_dataset.get_next_batch(batch_size, distribution)

    def next_val_batch(self, batch_size, distribution=None):
        return self.val_dataset.get_next_batch(batch_size, distribution)

if __name__ == '__main__':
    dataset = SliceBaseTumorDataset(Config)
    images, labels, bgs = dataset.next_train_batch(20)
    print labels
    images, labels, bgs = dataset.next_val_batch(20)
    print labels