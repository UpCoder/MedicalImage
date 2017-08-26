class Config:
    IMAGE_DIR = '/home/give/PycharmProjects/MedicalImage/imgs/LiverAndLesion_bg'
    TRAIN_IMAGE_DIR = '/home/give/PycharmProjects/MedicalImage/imgs/LiverAndLesion_bg/TRAIN'
    VALIDATION_IMAGE_DIR = '/home/give/PycharmProjects/MedicalImage/imgs/LiverAndLesion_bg/VAL'
    LESION_TYPE = ['CYST', 'FNH', 'HCC', 'HEM', 'METS']
    VALIDATION_DISTRIBUTION = [
        7,
        7,
        7,
        7,
        7
    ]