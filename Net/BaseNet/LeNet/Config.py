class Config():
    NEED_MUL = True
    MODEL_SAVE_PATH = '/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/model/'
    CONV_LAYERS_CONFIG = {
        'conv1_1': {
            'deep': 32,
            'size': 7,
            'dropout': True,
            'pooling': {
                'exists': True,
                'name': 'pooling1'
            },
            'batch_norm': True,
            'trainable': True
        },
        'conv2_1': {
            'deep': 32,
            'size': 5,
            'dropout': True,
            'pooling': {
                'exists': True,
                'name': 'pooling2'
            },
            'batch_norm': True,
            'trainable': True
        },
        'conv3_1': {
            'deep': 64,
            'size': 3,
            'dropout': True,
            'pooling': {
                'exists': True,
                'name': 'pooling3'
            },
            'batch_norm': True,
            'trainable': True
        }
        # 'conv3_1': {
        #     'deep': 32,
        #     'size': 3
        # },
    }

    FC_SIZE = 256

    BATCH_SIZE = 800

    OUTPUT_NODE = 5
    IMAGE_W = 64
    IMAGE_H = 64
    IMAGE_CHANNEL = 1

    REGULARIZTION_RATE = 1e-3
    MOVING_AVERAGE_DECAY = 0.997
    LEARNING_RATE = 1e-4

    DROP_OUT = True

    ITERATOE_NUMBER = int(1e+5)

    TRAIN_LOG_DIR = '/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/log/tumor_linear_enhancement/train'
    VAL_LOG_DIR = '/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/log/tumor_linear_enhancement/val'

    # IMAGE_W = 28
    # IMAGE_H = 28
    # IMAGE_CHANNEL = 1
    # OUTPUT_NODE = 10