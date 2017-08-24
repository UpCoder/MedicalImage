class Config():
    MODEL_SAVE_PATH = '/home/give/PycharmProjects/MedicalImage/Net/MyNet/MultiScaleNet/model/'
    OUTPUT_NODE = 5
    sizes = [
        [45, 45, 3],
        [20, 20, 3],
        [100, 100, 3]
    ]
    CONV_LAYERS_CONFIG = {
        'conv1_1': {
            'deep': 32,
            'size': 3,
            'dropout': True,
            'pooling': {
                'exists': True,
                'name': 'pooling1'
            }
        },
        'conv2_1': {
            'deep': 64,
            'size': 3,
            'dropout': True,
            'pooling': {
                'exists': True,
                'name': 'pooling2'
            }
        },
        'conv3_1': {
            'deep': 128,
            'size': 3,
            'dropout': True,
            'pooling': {
                'exists': True,
                'name': 'pooling3'
            }
        }
    }
    FC_LAYERS_CONFIG = {
        # 'fc1': {
        #     'size': 2048,
        #     'regularizer': True,
        #     'batch_norm': True
        # },
        'fc2': {
            'size': 256,
            'regularizer': True,
            'batch_norm': True
        },
        'fc3': {
            'size': OUTPUT_NODE,
            'regularizer': True,
            'batch_norm': False
        }
    }
    BATCH_SIZE = 50
    BATCH_DISTRIBUTION = [
        10,
        10,
        10,
        10,
        10,
    ]

    REGULARIZTION_RATE = 1e-1

    LEARNING_RATE = 1e-4

    DROP_OUT = True

    ITERATOE_NUMBER = int(1e+5)

    TRAIN_LOG_DIR = '/home/give/PycharmProjects/MedicalImage/Net/MyNet/MultiScaleNet/log/zero/train'
    VAL_LOG_DIR = '/home/give/PycharmProjects/MedicalImage/Net/MyNet/MultiScaleNet/log/zero/val'

    # IMAGE_W = 28
    # IMAGE_H = 28
    # IMAGE_CHANNEL = 1
    # OUTPUT_NODE = 10