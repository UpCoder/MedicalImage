class Config():
    NEED_MUL = False
    MODEL_SAVE_PATH = '/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/model/'
    CONV_LAYERS_CONFIG = {
        'conv1_1': {
            'deep': 32,
            'size': 3,
            'dropout': True,
            'pooling': {
                'max_pooling': True,
                'exists': True,
                'name': 'pooling1'
            }
        },
        'conv2_1': {
            'deep': 32,
            'size': 3,
            'dropout': True,
            'pooling': {
                'max_pooling': True,
                'exists': True,
                'name': 'pooling2'
            }
        },
        'conv3_1': {
            'deep': 64,
            'size': 3,
            'dropout': True,
            'pooling': {
                'max_pooling': True,
                'exists': True,
                'name': 'pooling3'
            }
        }
        # 'conv3_1': {
        #     'deep': 32,
        #     'size': 3
        # },
    }
    CONV_LAYERS_CONFIG_BG = {
        'conv1_1_bg': {
            'deep': 32,
            'size': 3,
            'dropout': True,
            'pooling': {
                'max_pooling': True,
                'exists': True,
                'name': 'pooling1_bg'
            }
        },
        'conv2_1_bg': {
            'deep': 32,
            'size': 3,
            'dropout': True,
            'pooling': {
                'max_pooling': True,
                'exists': True,
                'name': 'pooling2_bg'
            }
        },
        'conv3_1_bg': {
            'deep': 64,
            'size': 3,
            'dropout': True,
            'pooling': {
                'max_pooling': True,
                'exists': True,
                'name': 'pooling3_bg'
            }
        }
    }
    FC_SIZE = 512

    BATCH_SIZE = 40
    BATCH_DISTRIBUTION = [
        8,
        8,
        8,
        8,
        8,
    ]

    OUTPUT_NODE = 5
    IMAGE_W = 45
    IMAGE_H = 45
    BG_W = 60
    BG_H = 60
    IMAGE_CHANNEL = 3

    REGULARIZTION_RATE = 0.000000001

    LEARNING_RATE = 1e-4

    DROP_OUT = True

    ITERATOE_NUMBER = int(1e+4)

    TRAIN_LOG_DIR = '/home/give/PycharmProjects/MedicalImage/Net/MyNet/BG_LeNet/logs/tumor/train'
    VAL_LOG_DIR = '/home/give/PycharmProjects/MedicalImage/Net/MyNet/BG_LeNet/logs/tumor/val'

    # IMAGE_W = 28
    # IMAGE_H = 28
    # IMAGE_CHANNEL = 1
    # OUTPUT_NODE = 10