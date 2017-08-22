# -*- coding: utf-8 -*-


class Config:
    QIXIANGS = ['NC', 'ART', 'PV']
    LESION_TYPE = ['CYST', 'METS', 'HCC', 'HEM', 'FNH']
    LABEL_NUMBER_MAPPING = {
        'CYST': 0,
        'FNH': 1,
        'HCC': 2,
        'HEM': 3,
        'METS': 4,
    }
    ADJUST_WW_WC = True     # 判断是否需要调整窗窗位
    NPY_SAVE_PATH = 'E:\\work\\MedicalImage\\data'    # 存储的NPY文件的地址
    RECOVER = True
    BOUNDING_BOX_NOT_LEASION = 0    # 0代表补0 1代表补肝脏密度
    EXCEL_PATH = 'E:\\Resource\\DataSet\\MedicalImage\\data_origanal.xlsx'
    IMAGE_SAVE_PATH = 'E:\\work\\MedicalImage\\imgs'
    LESION_TYPE_RANGE = {
        'CYST': [range(0, 19), range(100, 120)],
        'FNH': [range(19, 29), range(120, 132)],
        'HCC': [range(29, 39), range(132, 152)],
        'HEM': [range(39, 49), range(152, 172)],
        'METS': [range(49, 59), range(172, 185)],
    }
    DATASET_PATH = 'E:\\Resource\\DataSet\\MedicalImage'
    MaxSliceDataPATH = [
        'E:\\work\\MedicalImage\\data\\MaxSlice_Image.npy',
        'E:\\work\\MedicalImage\\data\\MaxSlice_Mask.npy',
        'E:\\work\\MedicalImage\\data\\MaxSlice_Label.npy'
    ]
    MaxSlice_Resize = {
        'SAVE_PATH': [
            'E:\\work\\MedicalImage\\data\\MaxSlice_Resize_ROI.npy',
            'E:\\work\\MedicalImage\\data\\MaxSlice_Resize_Label.npy'
        ],
        'RESIZE': [
            45, 45
        ],
        'IMAGE_SAVE_PATH': 'E:\\work\\MedicalImage\\imgs\\resized'
    }