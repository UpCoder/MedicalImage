# -*- coding: utf-8 -*-
from inference import inference
import tensorflow as tf
from Config import Config as sub_Config
from Slice.MaxSlice.MaxSlice_Resize import MaxSlice_Resize
from tensorflow.examples.tutorials.mnist import input_data
from Tools import changed_shape, calculate_acc_error
import numpy as np
from Patch.PatchBase import PatchDataSet
from Patch.Config import Config as patch_config
from Slice.Non_WW_WC.SliceBase import SliceDataSet
from Slice.Non_WW_WC.Config import Config as slice_config


def train(dataset, load_model_path, save_model_path, train_log_dir, val_log_dir):
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
    # global_step = tf.Variable(0, trainable=False)
    # variable_average = tf.train.ExponentialMovingAverage(
    #     sub_Config.MOVING_AVERAGE_DECAY,
    #     global_step
    # )
    # variable_averages_op = variable_average.apply(tf.trainable_variables())
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
        loss=loss,
        # global_step=global_step
    )
    # with tf.control_dependencies([train_step, variable_averages_op]):
    #     train_op = tf.no_op(name='train')
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

        if load_model_path:
            saver.restore(sess, load_model_path)
        writer = tf.summary.FileWriter(train_log_dir, tf.get_default_graph())
        val_writer = tf.summary.FileWriter(val_log_dir, tf.get_default_graph())
        for i in range(sub_Config.ITERATOE_NUMBER):
            images, labels = dataset.get_next_train_batch(sub_Config.BATCH_SIZE)
            images = changed_shape(images, [
                    len(images),
                    sub_Config.IMAGE_W,
                    sub_Config.IMAGE_W,
                    sub_Config.IMAGE_CHANNEL
                ])
            if i == 0:
                from PIL import Image
                image = Image.fromarray(np.asarray(images[0, :, :, 0], np.uint8))
                image.show()
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
            if i % 500 == 0 and i != 0 and save_model_path is not None:
                # 保存模型
                save_model_path = save_model_path+'_' + str(i)
                import os
                if not os.path.exists(save_model_path):
                    os.mkdir(save_model_path)
                save_model_path += '/'
                saver.save(sess, save_model_path)
            if i % 100 == 0:
                validation_images, validation_labels = dataset.get_next_val_batch(sub_Config.BATCH_SIZE)
                validation_images = changed_shape(
                    validation_images,
                    [
                        len(validation_images),
                        sub_Config.IMAGE_W,
                        sub_Config.IMAGE_W,
                        sub_Config.IMAGE_CHANNEL
                    ]
                )
                validation_accuracy, validation_loss, summary, logits = sess.run(
                    [accuracy_tensor, loss, merge_op, y],
                    feed_dict={
                        x: validation_images,
                        y_: validation_labels
                    }
                )
                calculate_acc_error(
                    logits=np.argmax(logits, 1),
                    label=validation_labels,
                    show=True
                )
                val_writer.add_summary(summary, i)
                print 'step is %d,training loss value is %g,  accuracy is %g ' \
                      'validation loss value is %g, accuracy is %g' % \
                      (i, loss_value, accuracy_value, validation_loss, validation_accuracy)
        writer.close()
        val_writer.close()
if __name__ == '__main__':
    dataset = PatchDataSet(new_size=[sub_Config.IMAGE_W, sub_Config.IMAGE_H], config=patch_config)
    train(
        dataset,
        load_model_path=None,
        save_model_path='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/model/model_art',
        train_log_dir='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/log/patch_art/train',
        val_log_dir='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/log/patch_art/val'
    )