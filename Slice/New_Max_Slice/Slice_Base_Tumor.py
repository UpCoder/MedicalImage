# -*- coding: utf-8 -*-
# 主要是重写ｅｘｔｒａｃｔ_ｒｏｉ方法
# 该方法将提取病灶部分作为我们的ＲＯＩ
from Slice_Base import Slice_Base
from Config import Config
import numpy as np
from PIL import Image


class Slice_Base_Tumor(Slice_Base):
    def __init__(self, config):
        self.extract_roi_function = self.extract_roi_resize
        Slice_Base.__init__(self, config)

    def extract_roi(self):
        self.extract_roi_function()

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
if __name__ == '__main__':
    Slice_Base_Tumor(Config)