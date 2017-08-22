# -*- coding: utf-8 -*-
from Config import Config
import xlrd
from Tools import get_lesion_type_by_srrid
import os
import glob
import re

class ExcelData:
    def __init__(self):
        def convert_RS_int(rs_str):
            return int(rs_str[:rs_str.find('/')])
        self.data = xlrd.open_workbook(Config.EXCEL_PATH)
        table = self.data.sheets()[0]
        class_index = -1
        for index, name in enumerate(table.row_values(0)):
            if name == 'Class':
                class_index = index
        lesion_range1 = range(3, 101)
        lesion_range2 = range(106, 205)
        RS_NC_index = class_index + 2
        RS_ART_index = class_index + 3
        RS_PV_index = class_index + 4
        last_srrid = 'Srr000'
        lesions_by_lesion = {}
        lesions_by_srrid = {}
        count = 0
        for lesion_index in lesion_range1:
            srrid = table.row_values(lesion_index)[0]
            if not srrid.startswith('Srr'):
                srrid = last_srrid
            last_srrid = srrid
            lesion_type = get_lesion_type_by_srrid(int(srrid[3:]))
            pv_path = os.path.join(
                glob.glob(os.path.join(Config.DATASET_PATH, lesion_type, srrid[3:] + '-*'))[0],
                'PV'
            )
            dicom_file = os.listdir(pv_path)[0]
            re_pattern = '([0-9]*_[0-9]*_[0-9]*_[0-9]*)'
            re_res = re.match(re_pattern, dicom_file)
            use_str = re_res.group()
            start_index = int(use_str[-5:])
            if convert_RS_int(table.row_values(lesion_index)[RS_PV_index]) < start_index:
                start_index = 1
            slice_index = [
                int(srrid[3:]),
                convert_RS_int(table.row_values(lesion_index)[RS_NC_index]),
                convert_RS_int(table.row_values(lesion_index)[RS_ART_index]),
                convert_RS_int(table.row_values(lesion_index)[RS_PV_index]) - start_index + 1,
            ]
            lesion_type = lesion_type.replace(' ', '')
            if lesion_type in Config.LESION_TYPE:
                if lesion_type in lesions_by_lesion.keys():
                    lesions_by_lesion[lesion_type].append(slice_index)
                else:
                    lesions_by_lesion[lesion_type] = []
                    lesions_by_lesion[lesion_type].append(slice_index)
                if int(srrid[3:]) in lesions_by_srrid:
                    lesions_by_srrid[int(srrid[3:])].append(slice_index)
                else:
                    lesions_by_srrid[int(srrid[3:])] = []
                    lesions_by_srrid[int(srrid[3:])].append(slice_index)
            else:
                print 'error', lesion_type
        for lesion_index in lesion_range2:
            srrid = table.row_values(lesion_index)[0]
            if not srrid.startswith('Srr'):
                srrid = last_srrid
            last_srrid = srrid
            lesion_type = get_lesion_type_by_srrid(int(srrid[3:]))
            pv_path = os.path.join(
                glob.glob(os.path.join(Config.DATASET_PATH, lesion_type, srrid[3:] + '-*'))[0],
                'PV'
            )
            dicom_file = os.listdir(pv_path)[0]
            re_pattern = '([0-9]*_[0-9]*_[0-9]*_[0-9]*)'
            re_res = re.match(re_pattern, dicom_file)
            use_str = re_res.group()
            start_index = int(use_str[-5:])
            if convert_RS_int(table.row_values(lesion_index)[RS_PV_index]) < start_index:
                start_index = 1
            slice_index = [
                int(srrid[3:]),
                convert_RS_int(table.row_values(lesion_index)[RS_NC_index]),
                convert_RS_int(table.row_values(lesion_index)[RS_ART_index]),
                convert_RS_int(table.row_values(lesion_index)[RS_PV_index]) - start_index + 1,
            ]
            lesion_type = lesion_type.replace(' ', '')
            if lesion_type in Config.LESION_TYPE:
                if lesion_type in lesions_by_lesion.keys():
                    lesions_by_lesion[lesion_type].append(slice_index)
                else:
                    lesions_by_lesion[lesion_type] = []
                    lesions_by_lesion[lesion_type].append(slice_index)
                if int(srrid[3:]) in lesions_by_srrid:
                    lesions_by_srrid[int(srrid[3:])].append(slice_index)
                else:
                    lesions_by_srrid[int(srrid[3:])] = []
                    lesions_by_srrid[int(srrid[3:])].append(slice_index)
        self.lesions_by_lesion = lesions_by_lesion
        self.lesions_by_srrid = lesions_by_srrid

    @staticmethod
    def test_unit():
        count = 0
        count_map = {}
        lesions = ExcelData().lesions_by_lesion
        for key in lesions.keys():
            for lesion in lesions[key]:
                print lesion
                count += 1
                if key in count_map:
                    count_map[key] += 1
                else:
                    count_map[key] = 0
        print 'lesion number is ', count
        print count_map

        lesions = ExcelData().lesions_by_srrid
        count_map = {}
        count = 0
        for key in lesions.keys():
            print key
            for lesion in lesions[key]:
                print lesion
                count += 1
                if key in count_map:
                    count_map[key] += 1
                else:
                    count_map[key] = 1
        print 'lesion number is ', count
        print count_map[1]
if __name__ == '__main__':
    ExcelData.test_unit()