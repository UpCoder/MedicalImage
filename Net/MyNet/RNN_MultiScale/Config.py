# -*- coding: utf-8 -*-
class Config():
    MODEL_SAVE_PATH = '/home/give/PycharmProjects/MedicalImage/Net/MyNet/RNN_MultiScale/model/'
    OUTPUT_NODE = 5
    sizes = [
        [45, 45, 3],
        [20, 20, 3],
        [100, 100, 3]
    ]
    IMAGE_W = 45
    IMAGE_H = 45
    IMAGE_CHANNEL = 1
    NEED_MUL = True
    STATE_FEATURE_DIM = 2048    # 每个尺寸的特征向量的维度
    SIZE_FEATURE_DIM = 2048     # 每个尺寸的特征向量的维度
    CONV_LAYERS_CONFIG = {
        'conv1_1': {
            'deep': 32,
            'size': 3,
            'dropout': True,
            'pooling': {
                'exists': True,
                'name': 'pooling1',
                'is_max': True
            }
        },
        'conv2_1': {
            'deep': 64,
            'size': 3,
            'dropout': True,
            'pooling': {
                'exists': True,
                'name': 'pooling2',
                'is_max': True
            }
        },
        'conv3_1': {
            'deep': 128,
            'size': 3,
            'dropout': True,
            'pooling': {
                'exists': True,
                'name': 'pooling3',
                'is_max': True
            }
        }
    }
    FC_LAYERS_CONFIG = {
        'fc1': {
            'size': 512,
            'regularizer': True,
            'batch_norm': True
        },
        'fc2': {
            'size': 512,
            'regularizer': True,
            'batch_norm': True
        },
        'fc3': {
            'size': OUTPUT_NODE,
            'regularizer': True,
            'batch_norm': False
        }
    }
    BATCH_SIZE = 35
    BATCH_DISTRIBUTION = [
        7,
        7,
        7,
        7,
        7,
    ]

    REGULARIZTION_RATE = 1e-2

    LEARNING_RATE = 1e-4

    DROP_OUT = True

    ITERATOE_NUMBER = int(1e+4)

    TRAIN_LOG_DIR = '/home/give/PycharmProjects/MedicalImage/Net/MyNet/RNN_MultiScale/logs/linear_enhancement/train'
    VAL_LOG_DIR = '/home/give/PycharmProjects/MedicalImage/Net/MyNet/RNN_MultiScale/logs/linear_enhancement/val'

    # IMAGE_W = 28
    # IMAGE_H = 28
    # IMAGE_CHANNEL = 1
    # OUTPUT_NODE = 10