class Config:
    IMAGE_DIR = '/home/give/PycharmProjects/MedicalImage/imgs/LiverAndLesion'
    # TRAIN_IMAGE_DIR = '/home/give/PycharmProjects/MedicalImage/imgs/LiverAndLesion_bg/TRAIN'
    # VALIDATION_IMAGE_DIR = '/home/give/PycharmProjects/MedicalImage/imgs/LiverAndLesion_bg/VAL'
    TRAIN_IMAGE_DIR = '/home/give/PycharmProjects/MedicalImage/imgs/LiverAndLesionMultiSlice/train'
    VALIDATION_IMAGE_DIR = '/home/give/PycharmProjects/MedicalImage/imgs/LiverAndLesionMultiSlice/val'

    LESION_TYPE = ['CYST', 'FNH', 'HCC', 'HEM', 'METS']
    VALIDATION_DISTRIBUTION = [
        7,
        7,
        7,
        7,
        7
    ]