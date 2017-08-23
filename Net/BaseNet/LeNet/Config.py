class Config():
    CONV1_1_DEEP = 32
    CONV1_1_SIZE = 3

    CONV2_1_DEEP = 64
    CONV2_1_SIZE = 3

    FC_SIZE = 512

    BATCH_SIZE = 50
    BATCH_DISTRIBUTION = [
        10,
        10,
        10,
        10,
        10,
    ]

    OUTPUT_NODE = 5
    IMAGE_W = 45
    IMAGE_H = 45
    IMAGE_CHANNEL = 1

    LEARNING_RATE = 1e-4

    DROP_OUT = True

    ITERATOE_NUMBER = int(1e+5)

    LOG_DIR = '/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/log'

    # IMAGE_W = 28
    # IMAGE_H = 28
    # IMAGE_CHANNEL = 1
    # OUTPUT_NODE = 10