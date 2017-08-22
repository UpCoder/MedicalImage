# -*- coding: utf-8 -*-
import numpy as np
import os
from Config import Config
import scipy.io as scio
import gc
from ExcelData import ExcelData
import glob
from Tools import save_image, get_lesion_type_by_srrid, get_diff_phases_images, get_total_masks, save_image_with_mask


class MaxSlice_Base:
    def __init__(self, config):
        if os.path.exists(Config.MaxSliceDataPATH[0]):
            self.images = np.load(
                Config.MaxSliceDataPATH[0]
            )
            self.masks = np.load(
                Config.MaxSliceDataPATH[1]
            )
            self.labels = np.load(
                Config.MaxSliceDataPATH[2]
            )
            return
        self.lesions_data = ExcelData().lesions_by_srrid
        self.images, self.labels = MaxSlice_Base.load_images_labels(config, self.lesions_data)
        np.save(
            Config.MaxSliceDataPATH[0],
            self.images
        )
        np.save(
            Config.MaxSliceDataPATH[2],
            self.labels
        )
        del self.images, self.labels
        gc.collect()
        self.masks = MaxSlice_Base.load_masks(config, self.lesions_data)
        np.save(
            Config.MaxSliceDataPATH[1],
            self.masks
        )

    @staticmethod
    def cout_mask_size(config):
        mask_file_path = config.MaxSliceDataPATH[1]
        masks = np.load(mask_file_path)
        avg_y = 0.0
        avg_x = 0.0
        min_y = 9999999     # y方向上最小的差距
        max_y = 0.0     # y方向上最大的差距
        min_x = 9999999     # x方向上最小的差距
        max_x = 0.0     # x方向上最大的差距
        for index, phase_mask in enumerate(masks):
            print 'index is ', index
            for mask in phase_mask:
                [ys, xs] = np.where(mask != 0)
                if len(ys) == 0:
                    print 'Error'
                    continue
                miny = np.min(ys)
                maxy = np.max(ys)
                min_y = min(min_y, (maxy - miny))
                max_y = max(max_y, (maxy - miny))
                minx = np.min(xs)
                maxx = np.max(xs)
                min_x = min(min_x, (maxx - minx))
                max_x = max(max_x, (maxx - minx))
                avg_y += (maxy-miny)
                avg_x += (maxx-minx)
                print '(%d, %d)' % (maxy-miny, maxx-minx)
        print '(%f, %f)' % (avg_y/(len(masks)*3), avg_x/(len(masks)*3))
        print 'max is (%d, %d)' % (max_y, max_x)
        print 'min is (%d, %d)' % (min_y, min_x)

    @staticmethod
    def load_images_labels(config, lesions_data):
        image_slices = []
        mask_slices = []
        labels = []
        for key in lesions_data.keys():
            srrid = key
            str_srrid = '%03d' % srrid
            lesion_type = get_lesion_type_by_srrid(srrid)
            images_path = glob.glob(os.path.join(config.DATASET_PATH, lesion_type, str_srrid + '-*'))[0]
            images = get_diff_phases_images(images_path)
            masks = get_total_masks(images_path)
            tumors_mask = masks['TumorMask']
            print lesions_data[key]
            for index, lesion in enumerate(lesions_data[key]):
                nap_index = lesion[1:]
                print nap_index
                cur_image_slice = []
                if key != 45:
                    nc_image_slice = images['NC'][nap_index[0] - 1, :, :]
                    art_image_slice = images['ART'][nap_index[1] - 1, :, :]
                else:
                    nc_image_slice = images['NC'][np.shape(images['NC'])[0] - nap_index[0], :, :]
                    art_image_slice = images['ART'][np.shape(images['ART'])[0] - nap_index[1], :, :]
                if key in [45, 177]:
                    pv_image_slice = images['PV'][np.shape(images['PV'])[0] - nap_index[2], :, :]
                else:
                    pv_image_slice = images['PV'][nap_index[2] - 1, :, :]
                nc_mask_slice = tumors_mask[index]['NC'][np.shape(images['NC'])[0] - nap_index[0], :, :]
                art_mask_slice = tumors_mask[index]['ART'][np.shape(images['ART'])[0] - nap_index[1], :, :]
                pv_mask_slice = tumors_mask[index]['PV'][np.shape(images['PV'])[0] - nap_index[2], :, :]
                cur_image_slice.append(nc_image_slice)
                cur_image_slice.append(art_image_slice)
                cur_image_slice.append(pv_image_slice)
                image_slices.append(cur_image_slice)
                labels.append(Config.LABEL_NUMBER_MAPPING[lesion_type])
                save_image_with_mask(nc_image_slice, nc_mask_slice, os.path.join(config.IMAGE_SAVE_PATH, str(srrid) + '_' + str(index + 1) + '_nc_image_mask.jpg'))
                save_image_with_mask(art_image_slice, art_mask_slice, os.path.join(config.IMAGE_SAVE_PATH, str(srrid) + '_' + str(index + 1) + '_art_image_mask.jpg'))
                save_image_with_mask(pv_image_slice, pv_mask_slice, os.path.join(config.IMAGE_SAVE_PATH, str(srrid) + '_' + str(index + 1) + '_pv_image_mask.jpg'))
                # print key, nap_index
        return image_slices, labels

    @staticmethod
    def load_masks(config, lesions_data):
        image_slices = []
        mask_slices = []
        labels = []
        for key in lesions_data.keys():
            srrid = key
            str_srrid = '%03d' % srrid
            lesion_type = get_lesion_type_by_srrid(srrid)
            images_path = glob.glob(os.path.join(config.DATASET_PATH, lesion_type, str_srrid + '-*'))[0]
            images = get_diff_phases_images(images_path)
            masks = get_total_masks(images_path)
            tumors_mask = masks['TumorMask']
            print lesions_data[key]
            for index, lesion in enumerate(lesions_data[key]):
                nap_index = lesion[1:]
                print nap_index
                # print np.shape(images['NC']), np.shape(images['ART']), np.shape(images['PV'])[0]
                cur_mask_slice = []
                nc_mask_slice = tumors_mask[index]['NC'][np.shape(images['NC'])[0] - nap_index[0], :, :]
                art_mask_slice = tumors_mask[index]['ART'][np.shape(images['ART'])[0] - nap_index[1], :, :]
                pv_mask_slice = tumors_mask[index]['PV'][np.shape(images['PV'])[0] - nap_index[2], :, :]
                cur_mask_slice.append(nc_mask_slice)
                cur_mask_slice.append(art_mask_slice)
                cur_mask_slice.append(pv_mask_slice)
                mask_slices.append(cur_mask_slice)
                # print key, nap_index
        return mask_slices

    @staticmethod
    def test_unit():
        # MaxSlice_Base.cout_mask_size(Config)
        dataset = MaxSlice_Base(Config)
        print len(dataset.images)
if __name__ == '__main__':
    MaxSlice_Base.test_unit()