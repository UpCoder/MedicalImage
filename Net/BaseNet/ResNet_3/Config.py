class Config:
    MODEL_SAVE_PATH = '/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/model/'
    IMAGE_W = 64
    IMAGE_H = 64
    IMAGE_CHANNEL = 1
    phase_name = 'ART'
    MOMENTUM = 0.9
    OUTPUT_NODE = 5
    ITERATOE_NUMBER = int(1e+4)
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 100
    DISTRIBUTION = [
        20,
        20,
        20,
        20,
        20
    ]
