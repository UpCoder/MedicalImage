# -*- coding: utf-8 -*-
import os
from Tools import read_dicom_series, read_mhd_image, rejust_pixel_value, save_image
import numpy as np
import scipy.io as scio
import glob
import gc
from Config import Config

# UnitMedicalImage 对应的一次检查
class UnitMedicalImage:
    def __init__(self, image_path, lesion_type):
        self.image_path = image_path
        basename = os.path.basename(image_path)
        self.id = int(os.path.basename(image_path)[:basename.find('-')])
        save_npy_image_path = os.path.join(Config.NPY_SAVE_PATH, lesion_type + '_' + str(self.id) + '_image.mat')
        if os.path.exists(save_npy_image_path):
            return
        self.medical_image = UnitMedicalImage.load_image(self.image_path)
        self.mask_image = UnitMedicalImage.load_mask_image(self.image_path)
        scio.savemat(
            os.path.join(Config.NPY_SAVE_PATH, lesion_type + '_' + str(self.id) + '_image.mat'),
            self.medical_image
        )
        scio.savemat(
            os.path.join(Config.NPY_SAVE_PATH, lesion_type + '_' + str(self.id) + '_mask.mat'),
            self.mask_image
        )
        if Config.RECOVER:
            del self.medical_image, self.mask_image
            gc.collect()
    def save2image(self, save_path):
        for z in range(len(self.medical_image['ART'])):
            cur_slice = self.medical_image['ART'][z, :, :]
            save_image(cur_slice, os.path.join(save_path, str(z) + '.jpg'))

    @staticmethod
    def load_image(image_path):
        images = {}
        dirs = os.listdir(image_path)
        exists = False
        for dir_name in dirs:
            if dir_name in ['ART', 'NC', 'PV']:
                exists = True
                mhd_path = os.path.join(image_path, dir_name, dir_name+'.mhd')
                if os.path.exists(mhd_path):
                    images[dir_name] = read_mhd_image(mhd_path)
                else:
                    images[dir_name] = read_dicom_series(os.path.join(image_path, dir_name))
                if Config.ADJUST_WW_WC:
                    images[dir_name] = rejust_pixel_value(images[dir_name])
        if not exists:
            print image_path, 'happened error'
        return images

    @staticmethod
    def load_mask_image(image_path, mask_dir_name=['LiverMask', 'TumorMask']):
        mask_image = {}
        liver_mask = {}
        tumors_mask = []
        tumor_files_num = len(os.listdir(os.path.join(image_path, mask_dir_name[1]))) / 6
        for phase in Config.QIXIANGS:
            mask_file_name = glob.glob(os.path.join(image_path, mask_dir_name[0]) + '\\*'+phase+'.mhd')[0]
            liver_mask[phase] = read_mhd_image(mask_file_name)
        for i in range(tumor_files_num):
            tumor_mask_image = {}
            for phase in Config.QIXIANGS:
                tumor_mask_file_path = \
                glob.glob(os.path.join(image_path, mask_dir_name[1]) + '\\*' + phase + '_' + str(i + 1) + '.mhd')[0]
                tumor_mask_image[phase] = np.asarray(read_mhd_image(tumor_mask_file_path), np.uint8)
            tumors_mask.append(tumor_mask_image)
        mask_image['LiverMask'] = liver_mask
        mask_image['TumorMask'] = tumors_mask
        return mask_image

    @staticmethod
    def test_unit():
        check_path = 'E:\\Resource\\DataSet\\MedicalImage\\HCC\\135-8071855'
        unit_medical_image = UnitMedicalImage(check_path)
        print unit_medical_image.medical_image
        unit_medical_image.save2image('./imgs')


# OneLesionType 对应的是一种病灶类型
class OneLesionType:
    def __init__(self, lesion_path):
        self.lesion_path = lesion_path
        self.lesion_type = os.path.basename(self.lesion_path)
        self.medical_images = OneLesionType.load_lesion_images(self.lesion_path, self.lesion_type)

    @staticmethod
    def load_lesion_images(image_path, lesion_type):
        res = []
        dir_names = os.listdir(image_path)
        for dir_name in dir_names:
            if dir_name.startswith('delete'):
                continue
            dir_path = os.path.join(image_path, dir_name)
            print 'loading ', dir_path
            res.append(UnitMedicalImage(dir_path, lesion_type=lesion_type))
        return res

    @staticmethod
    def test_unit(lesion_path='E:\\Resource\\DataSet\\MedicalImage\\METS'):
        one_lesion = OneLesionType(lesion_path)
        print 'medical image number is ', len(one_lesion.medical_images)


class MedicalImageDataSet:
    def __init__(self, all_image_path):
        self.all_image_path = all_image_path
        self.all_medical_images = MedicalImageDataSet.load_medical_images(self.all_image_path)

    @staticmethod
    def load_medical_images(all_image_path):
        medical_images = {}
        lesion_type_names = os.listdir(all_image_path)
        for lesion_type_name in lesion_type_names:
            if lesion_type_name in Config.LESION_TYPE:
                medical_images[lesion_type_name] = OneLesionType(os.path.join(all_image_path, lesion_type_name)).medical_images
        return medical_images

    @staticmethod
    def test_unit(all_image_path='E:\\Resource\\DataSet\\MedicalImage'):
        all_images = MedicalImageDataSet(all_image_path)
        print all_images.all_image_path.keys()
if __name__ == '__main__':
    # OneLesionType.test_unit()
    # UnitMedicalImage.test_unit()
    MedicalImageDataSet.test_unit()

