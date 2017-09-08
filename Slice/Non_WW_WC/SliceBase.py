import os
from Config import Config
from Tools import shuffle_image_label, read_mhd_images
import glob


class SliceBase:
    def __init__(self, path, new_size):
        self.path = path
        self.new_size = new_size
        self.patchs_path = []
        self.labels = []
        self.images = []
        self.images, self.labels = self.load_images_label()
        self.images, self.labels = shuffle_image_label(self.images, self.labels)

    def load_images_label(self):
        images = []
        labels = []
        phases_name = ['NC', 'ART', 'PV']
        case_names = os.listdir(self.path)
        for case_name in case_names:
            labels.append(int(case_name[-1]))
            diff_phase_paths = [os.path.join(self.path, case_name, phase_name + '_ROI.mhd')
                                for phase_name in phases_name]
            diff_phase_images = read_mhd_images(diff_phase_paths, new_size=self.new_size)
            images.append(diff_phase_images)
        return images, labels

    def get_next_batch(self, batch_size, image_size):
        return self.images, self.labels


class SliceDataSet:
    def __init__(self, new_size, config=Config):
        self.train_dataset = SliceBase(config.TRAIN_DIR, new_size)
        self.val_dataset = SliceBase(config.VAL_DIR, new_size)
        self.new_size = new_size

    def get_next_train_batch(self, batch_size):
        return self.train_dataset.get_next_batch(batch_size, self.new_size)

    def get_next_val_batch(self, batch_size):
        return self.val_dataset.get_next_batch(batch_size, self.new_size)


if __name__ == '__main__':
    dataset = SliceDataSet([64, 64])
    images, labels = dataset.get_next_train_batch(
        20
    )
    import numpy as np

    print np.shape(images)