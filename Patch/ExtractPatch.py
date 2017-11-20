# -*- coding=utf-8 -*-
import os
from Tools import read_mhd_image, get_boundingbox, convert2depthlaster, show_image
import numpy as np
from glob import glob
from PIL import Image
import math
phasenames = ['NC', 'ART', 'PV']
class ExtractPatch:
    @staticmethod
    def extract_patch(dir_name, suffix_name, save_dir, patch_size, patch_step=1):
        '''
        提取指定类型病灶的ｐａｔｃｈ
        :param patch_size: 提取ｐａｔｃｈ的大小
        :param dir_name: 目前所有病例的存储路径
        :param suffix_name: 指定的病灶类型的后缀，比如说cyst 就是０
        :param save_dir:　提取得到的ｐａｔｃｈ的存储路径
        :param patch_step: 提取ｐａｔｃｈ的步长
        :return: None
        '''
        count = 0
        names = os.listdir(dir_name)
        for name in names:
            if name.endswith(suffix_name):
                # 只提取指定类型病灶的ｐａｔｃｈ
                mask_images = []
                mhd_images = []
                for phasename in phasenames:
                    image_path = glob(os.path.join(dir_name, name, phasename + '_Image*.mhd'))[0]
                    mask_path = os.path.join(dir_name, name, phasename + '_Registration.mhd')
                    mhd_image = read_mhd_image(image_path, rejust=True)
                    mhd_image = np.squeeze(mhd_image)
                    # show_image(mhd_image)
                    mask_image = read_mhd_image(mask_path)
                    mask_image = np.squeeze(mask_image)
                    [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)
                    # xmin -= 15
                    # xmax += 15
                    # ymin -= 15
                    # ymax += 15
                    mask_image = mask_image[xmin: xmax, ymin: ymax]
                    mhd_image = mhd_image[xmin: xmax, ymin: ymax]
                    mhd_image[mask_image != 1] = 0
                    mask_images.append(mask_image)
                    mhd_images.append(mhd_image)
                    # show_image(mhd_image)
                mask_images = convert2depthlaster(mask_images)
                mhd_images = convert2depthlaster(mhd_images)
                # show_image(mhd_images)
                count += 1
                [width, height, depth] = list(np.shape(mhd_images))
                patch_count = 1
                if width*height >= 400:
                    patch_step = int(math.sqrt(width*height/100))
                for i in range(patch_size/2, width - patch_size/2, patch_step):
                    for j in range(patch_size/2, height - patch_size/2, patch_step):
                        cur_patch = mhd_images[i-patch_size/2:i+patch_size/2, j-patch_size/2: j+patch_size/2, :]
                        if (np.sum(mask_images[i-patch_size/2:i+patch_size/2, j-patch_size/2: j+patch_size/2, :]) / ((patch_size-1) * (patch_size - 1) * 3)) < 0.9:
                            continue

                        save_path = os.path.join(save_dir, name+'_'+str(patch_count)+'.png')
                        patch_image = Image.fromarray(np.asarray(cur_patch, np.uint8))
                        patch_image.save(save_path)
                        patch_count += 1
                if patch_count == 1:
                    save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.png')
                    roi_image = Image.fromarray(np.asarray(mhd_images, np.uint8))
                    roi_image.save(save_path)

        print count


if __name__ == '__main__':
    for subclass in ['train', 'val']:
        for typeid in ['0', '1', '2', '3']:
            ExtractPatch.extract_patch(
                '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/'+subclass,
                typeid,
                '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/3phases/' + subclass + '/' + typeid,
                9
            )