# -*- coding: utf-8 -*-
from inference import inference
import tensorflow as tf
from Config import Config as sub_Config
from Slice.MaxSlice.MaxSlice_Resize import MaxSlice_Resize
from tensorflow.examples.tutorials.mnist import input_data
from Tools import changed_shape
import numpy as np


def train(dataset, load_model=False):
    x = tf.placeholder(
        tf.float32,
        shape=[
            None,
            sub_Config.IMAGE_W,
            sub_Config.IMAGE_H,
            sub_Config.IMAGE_CHANNEL
        ],
        name='input_x'
    )
    tf.summary.image(
        'input_x',
        x * 120,
        max_outputs=5
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
    y = inference(x, regularizer)
    tf.summary.histogram(
        'logits',
        tf.argmax(y, 1)
    )
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y,
            labels=tf.cast(y_, tf.int32)
        )
    ) + tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar(
        'loss',
        loss
    )
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

        if load_model:
            saver.restore(sess, sub_Config.MODEL_SAVE_PATH)
        writer = tf.summary.FileWriter(sub_Config.TRAIN_LOG_DIR, tf.get_default_graph())
        val_writer = tf.summary.FileWriter(sub_Config.VAL_LOG_DIR, tf.get_default_graph())
        for i in range(sub_Config.ITERATOE_NUMBER):
            # images, labels = dataset.train.next_batch(sub_Config.BATCH_SIZE)
            # labels = np.argmax(labels, 1)
            # # print np.shape(labels)
            # images = np.reshape(
            #     images,
            #     [
            #         sub_Config.BATCH_SIZE,
            #         sub_Config.IMAGE_W,
            #         sub_Config.IMAGE_H,
            #         sub_Config.IMAGE_CHANNEL
            #     ]
            # )
            images, labels = dataset.get_next_batch(sub_Config.BATCH_SIZE, sub_Config.BATCH_DISTRIBUTION)
            images = changed_shape(images, [
                    sub_Config.BATCH_SIZE,
                    sub_Config.IMAGE_W,
                    sub_Config.IMAGE_W,
                    3
                ])
            if i == 0:
                from PIL import Image
                image = Image.fromarray(np.asarray(images[0, :, :, 0], np.uint8))
                image.show()
            # images = np.reshape(
            #     images[:, :, :, 2],
            #     [
            #         sub_Config.BATCH_SIZE,
            #         sub_Config.IMAGE_W,
            #         sub_Config.IMAGE_W,
            #         1
            #     ]
            # )
            _, loss_value, accuracy_value, summary = sess.run(
                [train_op, loss, accuracy_tensor, merge_op],
                feed_dict={
                    x: images,
                    y_: labels
                }
            )
            writer.add_summary(
                summary=summary,
                global_step=i
            )
            if i % 1000 == 0 and i != 0:
                # 保存模型
                saver.save(sess, sub_Config.MODEL_SAVE_PATH)
            if i % 100 == 0:
                validation_images, validation_labels = dataset.get_validation_images_labels()
                validation_images = changed_shape(
                    validation_images,
                    [
                        len(validation_images),
                        sub_Config.IMAGE_W,
                        sub_Config.IMAGE_W,
                        3
                    ]
                )
                validation_accuracy, validation_loss, summary = sess.run(
                    [accuracy_tensor, loss, merge_op],
                    feed_dict={
                        x: validation_images,
                        y_: validation_labels
                    }
                )
                val_writer.add_summary(summary, i)
                print 'step is %d,training loss value is %g,  accuracy is %g ' \
                      'validation loss value is %g, accuracy is %g' % \
                      (i, loss_value, accuracy_value, validation_loss, validation_accuracy)
        writer.close()
        val_writer.close()
if __name__ == '__main__':
    dataset = MaxSlice_Resize(sub_Config)
    # mnist = input_data.read_data_sets("../data", one_hot=True)
    # train(mnist)