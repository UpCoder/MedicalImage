from Tools import read_mhd_image, show_image, save_image_with_mask, compress22dim
import os
from glob import glob
import numpy as np
data_dir = '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/train'
names = os.listdir(data_dir)
for name in names:
    cur_dir = os.path.join(data_dir, name)
    for phasename in ['NC', 'ART', 'PV']:
        image_path = glob(os.path.join(cur_dir, phasename + '_Image*.mhd'))[0]
        mask_paths = glob(os.path.join(cur_dir, phasename + '_Mask.mhd'))
        if len(mask_paths) == 0:
            mask_paths = glob(os.path.join(cur_dir, phasename + '_Mask_*.mhd'))
            for path in mask_paths:
                if path.endswith('Expand.mhd'):
                    continue
                else:
                    mask_path = path
                    break
        else:
            mask_path = mask_paths[0]
        mhd_image = read_mhd_image(image_path, rejust=True)
        mhd_image = compress22dim(mhd_image)
        print mask_path
        tumor_mask_image = read_mhd_image(mask_path)
        tumor_mask_image = compress22dim(tumor_mask_image)
        print np.max(tumor_mask_image)
        save_image_with_mask(mhd_image, tumor_mask_image, None)
    input('input %d')