class Config:
    MODEL_SAVE_PATH = '/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/model/'
    SIZES = [
        [64, 64],
        [128, 128]
    ]
    IMAGE_CHANNEL = 1
    phase_name='ART'
    ITERATOE_NUMBER = int(1e+4)
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 100
    OUTPUT_NODE = 2
    BATCH_DISTRIBUTION = [
        50,
        50,
    ]
