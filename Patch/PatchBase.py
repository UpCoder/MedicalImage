import os
from Config import Config
from Tools import shuffle_image_label, read_mhd_images
import glob


class PatchBase:
    def __init__(self, paths):
        self.paths = paths
        self.patchs_path = []
        self.labels = []
        for path in self.paths:
            cur_paths = PatchBase.load_paths(path)
            self.patchs_path.extend(cur_paths)
            self.labels.extend([int(os.path.basename(path))] * len(cur_paths))
        self.patchs_path, self.labels = shuffle_image_label(self.patchs_path, self.labels)
        self.startindex = 0
        self.epochnum = 0

    @staticmethod
    def load_paths(path):
        names = glob.glob(os.path.join(path, '*.mhd'))
        return names

    def get_next_batch(self, batch_size, image_size):
        end_index = self.startindex + batch_size
        if end_index > len(self.patchs_path):
            end_index = len(self.patchs_path)
        batch_path = self.patchs_path[self.startindex:end_index]
        batch_data = read_mhd_images(batch_path, image_size)
        batch_label = self.labels[self.startindex:end_index]
        if end_index == len(self.patchs_path):
            self.startindex = 0
            self.epochnum += 1
            print '-'*15, 'epoch num is ', self.epochnum, '-'*15
        else:
            self.startindex = end_index
        return batch_data, batch_label

class PatchDataSet:
    def __init__(self, new_size, config=Config):
        self.train_dataset = PatchBase(config.TRAIN_PATCH_PATHS)
        self.val_dataset = PatchBase(config.VAL_PATCH_PATHS)
        self.new_size = new_size
        
    def get_next_train_batch(self, batch_size):
        return self.train_dataset.get_next_batch(batch_size, self.new_size)

    def get_next_val_batch(self, batch_size):
        return self.val_dataset.get_next_batch(batch_size, self.new_size)

if __name__ == '__main__':
    dataset = PatchDataSet([64, 64])
    images, labels = dataset.get_next_train_batch(
        20
    )
    import numpy as np
    print np.shape(images)