class Config():
    NEED_MUL = True
    MODEL_SAVE_PATH = '/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/model/'
    CONV_LAYERS_CONFIG = {
        'NC': {
            'CONV1_1_NC': {
                'deep': 32,
                'size': 5,
                'dropout': True,
                'pooling': {
                    'exists': True,
                    'name': 'pooling1'
                },
                'batch_norm': True,
                'trainable': True
            },
            'CONV2_1_NC': {
                'deep': 64,
                'size': 5,
                'dropout': True,
                'pooling': {
                    'exists': True,
                    'name': 'pooling2'
                },
                'batch_norm': True,
                'trainable': True
            },
            # 'CONV3_1_NC': {
            #     'deep': 64,
            #     'size': 3,
            #     'dropout': True,
            #     'pooling': {
            #         'exists': True,
            #         'name': 'pooling3'
            #     },
            #     'batch_norm': True,
            #     'trainable': True
            # }
        },
        'ART': {
            'CONV1_1_ART': {
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
            'CONV2_1_ART': {
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
            'CONV3_1_ART': {
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
        },
        'PV': {
            'CONV1_1_PV': {
                'deep': 32,
                'size': 3,
                'dropout': True,
                'pooling': {
                    'exists': True,
                    'name': 'pooling1'
                },
                'batch_norm': True,
                'trainable': True
            },
            'CONV2_1_PV': {
                'deep': 64,
                'size': 3,
                'dropout': True,
                'pooling': {
                    'exists': True,
                    'name': 'pooling2'
                },
                'batch_norm': True,
                'trainable': True
            },
            # 'CONV3_1_PV': {
            #     'deep': 64,
            #     'size': 3,
            #     'dropout': True,
            #     'pooling': {
            #         'exists': True,
            #         'name': 'pooling3'
            #     },
            #     'batch_norm': True,
            #     'trainable': True
            # }
        }
    }

    FC_SIZE = 256

    BATCH_SIZE = 800

    OUTPUT_NODE = 5
    IMAGE_W = 64
    IMAGE_H = 64
    IMAGE_CHANNEL = 1

    REGULARIZTION_RATE = 1e-3
    MOVING_AVERAGE_DECAY = 0.9997
    LEARNING_RATE = 1e-4

    DROP_OUT = True

    ITERATOE_NUMBER = int(1e+5)

    TRAIN_LOG_DIR = '/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/log/tumor_linear_enhancement/train'
    VAL_LOG_DIR = '/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/log/tumor_linear_enhancement/val'

    # IMAGE_W = 28
    # IMAGE_H = 28
    # IMAGE_CHANNEL = 1
    # OUTPUT_NODE = 10