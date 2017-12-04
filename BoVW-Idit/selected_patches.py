import os
patches_num = 50000
from Tools import shuffle_array
import shutil

def selected_patches(source_dir, target_dir):
    names = os.listdir(source_dir)
    names = shuffle_array(names)
    names = names[:patches_num]
    paths = [os.path.join(source_dir, name) for name in names]
    for path in paths:
        shutil.copy(path, os.path.join(target_dir, os.path.basename(path)))

def selected_patches_multidir():
    data_dir = '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/3phase_npy_nonlimited'
    target_dir = '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/3phase_npy_nonlimited/balance'
    for subclass in ['train', 'val']:
        for typeid in [0, 1, 2, 3]:
            selected_patches(
                os.path.join(data_dir, subclass, str(typeid)),
                os.path.join(target_dir, subclass, str(typeid))
            )

if __name__ == '__main__':
    selected_patches_multidir()