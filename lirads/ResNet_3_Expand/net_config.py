class Config:
    MODEL_SAVE_PATH = '/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/model/'
    ROI_SIZE_W = 64
    ROI_SIZE_H = 64
    EXPAND_SIZE_W = 256
    EXPAND_SIZE_H = 256
    IMAGE_CHANNEL = 1
    phase_name = 'ART'
    MOMENTUM = 0.9
    OUTPUT_NODE = 2
    ITERATOE_NUMBER = int(1e+4)
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 100
    DISTRIBUTION = [
        50,
        50,
    ]
