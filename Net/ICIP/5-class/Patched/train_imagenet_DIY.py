# -*- coding=utf-8 -*-
from resnet_train_DIY import train
import tensorflow as tf
import time
import os
import sys
import re
import numpy as np
from resnet import inference_small
from image_processing import image_preprocessing
from Net.forpatch.ResNetMultiPhaseExpand.Config import Config as net_config
from Tools import shuffle_image_label

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', net_config.BATCH_SIZE, "batch size")

def main(_):
    roi_images = tf.placeholder(
        shape=[
            net_config.BATCH_SIZE,
            net_config.ROI_SIZE_W,
            net_config.ROI_SIZE_H,
            net_config.IMAGE_CHANNEL
        ],
        dtype=np.float32,
        name='roi_input'
    )
    expand_roi_images = tf.placeholder(
        shape=[
            net_config.BATCH_SIZE,
            net_config.EXPAND_SIZE_W,
            net_config.EXPAND_SIZE_H,
            net_config.IMAGE_CHANNEL
        ],
        dtype=np.float32,
        name='expand_roi_input'
    )
    labels_tensor = tf.placeholder(
        shape=[
            None
        ],
        dtype=np.int32
    )
    is_training_tensor = tf.placeholder(dtype=tf.bool, shape=[])
    logits = inference_small(
        roi_images,
        phase_names=['NC', 'ART', 'PV'],
        num_classes=5,
        is_training=is_training_tensor,
        point_phase=[2]
        )
    save_model_path = '/home/give/PycharmProjects/MedicalImage/Net/ICIP/Patched/models'
    train(logits, roi_images, labels_tensor, is_training_tensor, save_model_path=save_model_path, step_width=100)

if __name__ == '__main__':
    tf.app.run()
