class Config():
    MODEL_SAVE_PATH = '/home/give/PycharmProjects/MedicalImage/Net/MyNet/MultiScaleNet/model/'
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
            'deep': 32,
            'size': 3,
            'dropout': True,
            'pooling': {
                'exists': True,
                'name': 'pooling2'
            }
        },
        'conv3_1': {
            'deep': 64,
            'size': 3,
            'dropout': True,
            'pooling': {
                'exists': True,
                'name': 'pooling3'
            }
        }
        # 'conv3_1': {
        #     'deep': 32,
        #     'size': 3
        # },
    }

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
    sizes = [
        [45, 45, 3],
        [20, 20, 3],
        [100, 100, 3]
    ]

    REGULARIZTION_RATE = 1e-0

    LEARNING_RATE = 1e-4

    DROP_OUT = True

    ITERATOE_NUMBER = int(1e+5)

    LOG_DIR = '/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/log'

    # IMAGE_W = 28
    # IMAGE_H = 28
    # IMAGE_CHANNEL = 1
    # OUTPUT_NODE = 10