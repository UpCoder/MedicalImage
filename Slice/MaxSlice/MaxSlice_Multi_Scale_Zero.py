# -*- coding: utf-8 -*-
from MaxSlice_Base import MaxSlice_Base
import numpy as np
from PIL import Image
from Config import Config
from Tools import mark_outer_zero
import os
import glob


class MaxSlice_Multi_Scale_Zero(MaxSlice_Base):
    def __init__(self, config):
        self.config = config
        MaxSlice_Base.__init__(self, config)
        self.images, self.masks, self.labels = MaxSlice_Base.load_image_mask_label(config)
        self.roi_images = MaxSlice_Multi_Scale_Zero.load_roi_images(self.config.MaxSlice_Multi_Scale_Zero['IMAGE_SAVE_PATH'], self.config)
        # 首先我们需要生成数据，执行下面的函数，注释掉上面的函数
        # self.roi_images = MaxSlice_Multi_Scale_Zero.get_all_scale_images_masks(
        #     self.images,
        #     self.masks,
        #     self.config.MaxSlice_Multi_Scale['sizes'],
        #     self.config
        # )
        self.roi_images = np.array(self.roi_images)
        self.start_index = 0
        self.epoch_num = 0
        self.labels = np.array(self.labels)
        self.shuffle_ROI()
        self.split_train_and_validation()
        self.save_diff_scale()

    def save_diff_scale(self):
        for index in range(len(self.config.MaxSlice_Multi_Scale_Zero['sizes'])):
            cur_images = self.roi_images[:, index, :]
            print self.config.MaxSlice_Multi_Scale_Zero['NPY_SAVE_PATH'][index][0]
            np.save(self.config.MaxSlice_Multi_Scale_Zero['NPY_SAVE_PATH'][index][0], cur_images)


    @staticmethod
    def load_roi_images(IMAGE_DIR, config):
        image_names = list(os.listdir(IMAGE_DIR))
        image_names.sort()
        images = []
        flag = {}
        for image_name in image_names:
            index = int(image_name[:image_name.find('_')])
            if index not in flag.keys():
                flag[index] = 1
            else:
                continue
            diff_scale_phase_images = []
            for cur_size in config.MaxSlice_Multi_Scale_Zero['sizes']:
                diff_phase_images = []
                diff_phase_glob = os.path.join(IMAGE_DIR, str(index) + '_*_'+str(cur_size[0])+'.jpg')
                diff_phase_files = list(glob.glob(diff_phase_glob))
                # print 'phase glob is ', diff_phase_glob
                # print 'phase files is ', diff_phase_files
                diff_phase_files.sort()     # 确定都是从ＮＣphase开始的
                for diff_phase_file in diff_phase_files:
                    image = Image.open(diff_phase_file)
                    # print 'shape is ', np.shape(image)
                    diff_phase_images.append(np.array(image))
                diff_scale_phase_images.append(diff_phase_images)
            images.append(diff_scale_phase_images)
        print 'load roi image finish, it\'s length is ', len(images)
        return images

    @staticmethod
    def get_all_scale_images_masks(images, masks, sizes, config):
        res_images = []
        for index, image in enumerate(images):
            mask = masks[index]
            cur_images = []
            for size in sizes:
                cur_image = MaxSlice_Multi_Scale_Zero.resize_images([image], [mask], size, index, config)
                cur_images.append(cur_image)
            res_images.append(cur_images)
        return res_images

    @staticmethod
    def resize_images(images, masks, new_size, cur_index, config):
        roi_images = []
        print np.shape(images)
        print np.shape(masks)
        for index, phase_images in enumerate(images):
            print index
            cur_roi_images = []
            for phase_index, image in enumerate(phase_images):
                mask_image = masks[index][phase_index]
                image = np.array(image)
                [ys, xs] = np.where(mask_image != 0)
                miny = np.min(ys)
                maxy = np.max(ys)
                minx = np.min(xs)
                maxx = np.max(xs)
                if index != 145:
                    image, _ = mark_outer_zero(image, mask_image)
                else:
                    image[np.where(mask_image == 0)] = 0
                ROI = image[miny - 2:maxy + 2, minx - 2:maxx + 2]
                ROI_Image = Image.fromarray(np.asarray(ROI, np.uint8))
                ROI_Image = ROI_Image.resize(new_size)
                ROI_Image.save(
                    os.path.join(config.MaxSlice_Multi_Scale_Zero['IMAGE_SAVE_PATH'], str(cur_index) + '_' + str(phase_index) + '_' + str(new_size[0]) + '.jpg')
                )
                cur_roi_images.append(np.array(ROI_Image))
            roi_images.append(cur_roi_images)
        return roi_images

    @staticmethod
    def test_unit():
        # dataset = MaxSlice_Multi_Scale_Zero(Config)
        # for per_image in dataset.roi_images:
        #     for per_size in per_image:
        #         print 'resize result is ', np.shape(per_size)
        # print len(dataset.roi_images)
        images = MaxSlice_Multi_Scale_Zero.load_roi_images('/home/give/PycharmProjects/MedicalImage/imgs/resize_zero_multi_scale', Config)
        for diff_scale_phase_images in images:
            print np.shape(diff_scale_phase_images)
            for diff_phase_images in diff_scale_phase_images:
                print np.shape(diff_phase_images)

if __name__ == '__main__':
    MaxSlice_Multi_Scale_Zero.test_unit()