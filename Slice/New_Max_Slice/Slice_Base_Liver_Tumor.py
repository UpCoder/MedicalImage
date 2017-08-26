# -*- coding: utf-8 -*-
# 主要是重写ｅｘｔｒａｃｔ_ｒｏｉ方法
# 将肝脏和病灶相结合，作为我们的ＲＯＩ
from Slice_Base import Slice_Base
from Config import Config
import numpy as np
from PIL import Image
from Tools import show_image


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
        print 'linear rate is ', x
        new_tumor = tumor_image * x
        # show_image(new_tumor, 'after')
        return new_tumor

class Slice_Base_Liver_Tumor(Slice_Base):
    def __init__(self, config, operation, size=[256, 256]):
        # self.extract_roi_function = self.extract_roi_resize
        self.extract_roi_function = self.extract_roi_diff_size
        self.do_operation_liver_tumor = operation
        self.single_size = size
        Slice_Base.__init__(self, config)

    def extract_roi(self):
        self.extract_roi_function()

    def extract_roi_resize(self, new_size):
        self.rois = []
        for index, diff_phase_liver in enumerate(self.livers):
            new_diff_phase_lession = []
            print 'label is ', self.labels[index]
            for index_phase, single_phase_liver in enumerate(diff_phase_liver):
                single_phase_lession = self.do_operation_liver_tumor(single_phase_liver, self.lesions[index][index_phase])
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
                    single_phase_lesion = self.do_operation_liver_tumor(single_phase_liver,
                                                                         self.lesions[index][index_phase])
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

    @staticmethod
    def unit_test():
        dataset = Slice_Base_Liver_Tumor(Config)
        for diff_size_phase_lesions in dataset.rois:
            print np.shape(diff_size_phase_lesions[2][0])

if __name__ == '__main__':
    # Slice_Base_Liver_Tumor(Config).unit_test()
    liver_image_path = '/home/give/PycharmProjects/MedicalImage/imgs/LiverAndLesion/50_0_4/liver_art.jpg'
    mask_image_path = '/home/give/PycharmProjects/MedicalImage/imgs/LiverAndLesion/50_0_4/tumor_art.jpg'
    Liver_Tumor_Operations.tumor_linear_enhancement(
        np.array(Image.open(liver_image_path)),
        np.array(Image.open(mask_image_path))
    )