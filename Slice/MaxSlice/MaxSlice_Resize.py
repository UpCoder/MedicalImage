from Slice.MaxSlice.MaxSlice_Base import MaxSlice_Base
import numpy as np
from PIL import Image
from Config import Config
from Tools import save_image
import os

class MaxSlice_Resize(MaxSlice_Base):
    def __init__(self, config):
        MaxSlice_Base.__init__(self, config)
        self.roi_images = MaxSlice_Resize.resize_images(self.images, self.masks, config.MaxSlice_Resize['RESIZE'])
        print np.shape(self.roi_images)
        for index, roi_images_phase in enumerate(self.images):
            for phase_index, roi_image in enumerate(roi_images_phase):
                save_image(roi_image, os.path.join(config.MaxSlice_Resize['IMAGE_SAVE_PATH'], str(index) + ' _ ' + str(phase_index) + '.jpg'))
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
        MaxSlice_Resize(Config)

if __name__ == '__main__':
    MaxSlice_Resize.test_unit()
