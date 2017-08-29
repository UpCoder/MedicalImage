# -*- coding: utf-8 -*-
from inference import inference
import tensorflow as tf
from Config import Config as sub_Config
from Slice.MaxSlice.MaxSlice_Resize import MaxSlice_Resize
from Tools import changed_shape, calculate_acc_error
import numpy as np
from param_shared import inference as inference_parllel


def train(dataset):
    x1 = tf.placeholder(
        tf.float32,
        shape=[
            sub_Config.BATCH_SIZE,
            sub_Config.sizes[0][0],
            sub_Config.sizes[0][1],
            sub_Config.sizes[0][2]
        ],
        name='input_x1'
    )
    x2 = tf.placeholder(
        tf.float32,
        shape=[
            sub_Config.BATCH_SIZE,
            sub_Config.sizes[1][0],
            sub_Config.sizes[1][1],
            sub_Config.sizes[1][2]
        ],
        name='input_x2'
    )
    x3 = tf.placeholder(
        tf.float32,
        shape=[
            sub_Config.BATCH_SIZE,
            sub_Config.sizes[2][0],
            sub_Config.sizes[2][1],
            sub_Config.sizes[2][2]
        ],
        name='input_x3'
    )
    y_ = tf.placeholder(
        tf.float32,
        shape=[
            None,
        ]
    )
    tf.summary.histogram(
        'label',
        y_
    )
    regularizer = tf.contrib.layers.l2_regularizer(sub_Config.REGULARIZTION_RATE)
    y = inference_parllel([x1, x2, x3], regularizer)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y,
            labels=tf.cast(y_, tf.int32)
        )
    ) + tf.add_n(tf.get_collection('losses'))
    train_op = tf.train.GradientDescentOptimizer(
        learning_rate=sub_Config.LEARNING_RATE
    ).minimize(
        loss=loss
    )
    with tf.variable_scope('accuracy'):
        accuracy_tensor = tf.reduce_mean(
            tf.cast(
                tf.equal(x=tf.argmax(y, 1), y=tf.cast(y_, tf.int64)),
                tf.float32
            )
        )
        tf.summary.scalar(
            'accuracy',
            accuracy_tensor
        )
    saver = tf.train.Saver()
    merge_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, sub_Config.MODEL_SAVE_PATH)     # 加载模型
        validation_images, validation_labels = dataset.get_validation_images_labels()
        images1 = validation_images[:, 0, :]
        images2 = validation_images[:, 1, :]
        images3 = validation_images[:, 2, :]
        images1 = changed_shape(images1, [
            len(validation_images),
            sub_Config.sizes[0][0],
            sub_Config.sizes[0][1],
            sub_Config.sizes[0][2]
        ])
        images2 = changed_shape(images2, [
            len(validation_images),
            sub_Config.sizes[1][0],
            sub_Config.sizes[1][1],
            sub_Config.sizes[1][2]
        ])
        images3 = changed_shape(images3, [
            len(validation_images),
            sub_Config.sizes[2][0],
            sub_Config.sizes[2][1],
            sub_Config.sizes[2][2]
        ])
        validation_accuracy, validation_loss, summary, logits = sess.run(
            [accuracy_tensor, loss, merge_op, y],
            feed_dict={
                x1: images1,
                x2: images2,
                x3: images3,
                y_: validation_labels
            }
        )
        calculate_acc_error(
            logits=np.argmax(logits, 1),
            label=validation_labels,
            show=True
        )

if __name__ == '__main__':
    dataset = MaxSlice_Resize(sub_Config)
    # mnist = input_data.read_data_sets("../data", one_hot=True)
    # train(mnist)