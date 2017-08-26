# -*- coding: utf-8 -*-
# 主要是重写ｅｘｔｒａｃｔ_ｒｏｉ方法
# 该方法将提取病灶层对应的肝脏层作为我们的ＲＯＩ
from Slice_Base import Slice_Base
from Config import Config
import numpy as np
from PIL import Image
from Tools import show_image


class Slice_Base_Liver(Slice_Base):
    def __init__(self, config):
        # self.extract_roi_function = self.extract_roi_resize
        self.extract_roi_function = self.extract_roi_diff_size
        Slice_Base.__init__(self, config)

    def extract_roi(self):
        self.extract_roi_function()

    def extract_roi_resize(self, new_size=[256, 256]):
        self.rois = []
        average_w = 0.0
        average_h = 0.0
        min_w = 999999
        max_w = 0.0
        max_h = 0.0
        min_h = 999999
        for diff_phase_lession in self.livers:
            new_diff_phase_lession = []
            for single_phase_lession in diff_phase_lession:
                [xs, ys] = np.where(single_phase_lession != 0)
                min_xs = np.min(xs)
                max_xs = np.max(xs)
                min_ys = np.min(ys)
                max_ys = np.max(ys)
                min_w = min(min_w, max_xs - min_xs)
                max_w = max(max_w, max_xs - min_xs)
                min_h = min(min_h, max_ys - min_ys)
                max_h = max(max_h, max_ys - min_ys)
                average_w += (max_xs - min_xs)
                average_h += (max_ys - min_ys)
                single_phase_lession = single_phase_lession[min_xs:max_xs+1, min_ys:max_ys+1]
                image = Image.fromarray(single_phase_lession)
                image = image.resize(new_size)
                new_diff_phase_lession.append(
                    np.array(image)
                )
            self.rois.append(new_diff_phase_lession)
        self.rois = np.array(self.rois)
        print 'average w is %g, average h is %g' % (average_w / (3.0 * len(self.rois)), average_h / (3.0 * len(self.rois)))
        print 'min w is %d, min h is %d' % (min_w, min_h)
        print 'max w is %d, max h is %d' % (max_w, max_h)

    def extract_roi_diff_size(self, new_sizes=[[200, 200], [130, 100], [250, 300]]):
        self.rois = []
        for diff_phase_lesion in self.livers:
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

    @staticmethod
    def unit_test():
        dataset = Slice_Base_Liver(Config)
        for diff_size_phase_lesions in dataset.rois:
            print np.shape(diff_size_phase_lesions[2][0])

if __name__ == '__main__':
    Slice_Base_Liver(Config).unit_test()