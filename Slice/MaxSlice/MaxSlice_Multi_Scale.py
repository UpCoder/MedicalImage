from MaxSlice_Base import MaxSlice_Base
import numpy as np
from PIL import Image
from Config import Config


class MaxSlice_Multi_Scale(MaxSlice_Base):
    def __init__(self, config):
        self.config = config
        MaxSlice_Base.__init__(self, config)
        self.images, self.masks, self.labels = MaxSlice_Base.load_image_mask_label(config)
        self.roi_images = MaxSlice_Multi_Scale.get_all_scale_images_masks(
            self.images,
            self.masks,
            self.config.MaxSlice_Multi_Scale['sizes']
        )
        self.roi_images = np.array(self.roi_images)
        self.start_index = 0
        self.epoch_num = 0
        self.labels = np.array(self.labels)
        self.shuffle_ROI()
        self.split_train_and_validation()

    @staticmethod
    def get_all_scale_images_masks(images, masks, sizes):
        res_images = []
        for index, image in enumerate(images):
            mask = masks[index]
            cur_images = []
            for size in sizes:
                cur_image = MaxSlice_Multi_Scale.resize_images([image], [mask], size)
                cur_images.append(cur_image)
            res_images.append(cur_images)
        return res_images

    @staticmethod
    def resize_images(images, masks, new_size):
        roi_images = []
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
            roi_images.extend(np.array(cur_roi_images))
        return roi_images

    @staticmethod
    def test_unit():
        dataset = MaxSlice_Multi_Scale(Config)
        for per_image in dataset.roi_images:
            for per_size in per_image:
                print 'resize result is ', np.shape(per_size)
        print len(dataset.roi_images)


if __name__ == '__main__':
    MaxSlice_Multi_Scale.test_unit()