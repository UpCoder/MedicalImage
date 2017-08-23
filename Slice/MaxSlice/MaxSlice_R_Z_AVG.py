# -*- coding: utf-8 -*-
from MaxSlice_Base import MaxSlice_Base
import numpy as np
from PIL import Image
from Config import Config


# 对每个phase，我们减去数据集中该phase的平均值
class MaxSlice_R_Z_AVG(MaxSlice_Base):
    def __init__(self, config):
        self.config = config
        MaxSlice_Base.__init__(self, self.config)
        self.images, self.masks, self.labels = MaxSlice_Base.load_image_mask_label(config)
        print 'load images shape is ', np.shape(self.images)
        self.roi_images = MaxSlice_R_Z_AVG.resize_images(self.images, self.masks, config.MaxSlice_Resize['RESIZE'])
        self.sub_phase_average()    # 减去平均值
        print np.shape(self.roi_images)
        self.save_ROI_image(self.config.MaxSlice_R_Z_AVG['IMAGE_SAVE_PATH'])
        self.start_index = 0
        self.epoch_num = 0
        self.roi_images = np.array(self.roi_images)
        self.labels = np.array(self.labels)
        # self.shuffle_ROI()
        self.split_train_and_validation()

    # 针对我们的ＲＯＩＩｍａｇｅ　减去平均值
    def sub_phase_average(self):
        self.roi_images = np.array(self.roi_images)
        for i in range(len(self.config.QIXIANGS)):
            phase_reduce_mean = np.mean(self.roi_images[:, i, :, :])
            print 'phase index is %d, average is %g' % (i, phase_reduce_mean)
            self.roi_images[:, i, :, :] = self.roi_images[:, i, :, :] - phase_reduce_mean

    @staticmethod
    def resize_images(images, masks, new_size):
        roi_images = []
        print np.shape(images)
        print np.shape(masks)
        for index, phase_images in enumerate(images):
            cur_roi_images = []
            for phase_index, image in enumerate(phase_images):
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
        dataset = MaxSlice_R_Z_AVG(Config)

if __name__ == '__main__':
    MaxSlice_R_Z_AVG.test_unit()