# -*- coding: utf-8 -*-
# 使用ｐａｔｃｈ训练好的模型，来对ＲＯＩ进行微调
from resnet import inference_small, loss
import tensorflow as tf
from Config import Config as sub_Config
from Tools import changed_shape, calculate_acc_error, acc_binary_acc
import numpy as np
from Patch.ValDataMultiSize import ValDataSet


def train(train_data_set, val_data_set, load_model_path, save_model_path):
    xs = []
    for index in range(len(sub_Config.SIZES)):
        xs.append(
            tf.placeholder(
                tf.float32,
                shape=[
                    None,
                    sub_Config.SIZES[index][0],
                    sub_Config.SIZES[index][1]
                ]
            )
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
    global_step = tf.Variable(0, trainable=False)
    # variable_average = tf.train.ExponentialMovingAverage(
    #     sub_Config.MOVING_AVERAGE_DECAY,
    #     global_step
    # )
    # vaeriable_average_op = variable_average.apply(tf.trainable_variables())
    # regularizer = tf.contrib.layers.l2_regularizer(sub_Config.REGULARIZTION_RATE)
    is_training = tf.placeholder('bool', [], name='is_training')
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar-data',
                               'where to store the dataset')
    tf.app.flags.DEFINE_boolean('use_bn', True, 'use batch normalization. otherwise use biases')
    y = inference_small(xs, is_training=is_training,
                        num_classes=sub_Config.OUTPUT_NODE,
                        use_bias=FLAGS.use_bn,
                        num_blocks=3)
    tf.summary.histogram(
        'logits',
        tf.argmax(y, 1)
    )
    loss_ = loss(
        logits=y,
        labels=tf.cast(y_, np.int32)

    )
    tf.summary.scalar(
        'loss',
        loss_
    )
    train_op = tf.train.GradientDescentOptimizer(
        learning_rate=sub_Config.LEARNING_RATE
    ).minimize(
        loss=loss_,
        global_step=global_step
    )
    # with tf.control_dependencies([train_step, vaeriable_average_op]):
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
        writer = tf.summary.FileWriter('./log/fine_tuning/train', tf.get_default_graph())
        val_writer = tf.summary.FileWriter('./log/fine_tuning/val', tf.get_default_graph())
        for i in range(sub_Config.ITERATOE_NUMBER):
            images, labels = train_data_set.get_next_batch(sub_Config.BATCH_SIZE, sub_Config.BATCH_DISTRIBUTION)
            feed_dict = {}
            for index in range(len(sub_Config.SIZES)):
                feed_dict.setdefault(xs[index], images[index])
            feed_dict.setdefault(y_, labels)
            _, loss_value, accuracy_value, summary, global_step_value = sess.run(
                [train_op, loss_, accuracy_tensor, merge_op, global_step],
                feed_dict=feed_dict
            )
            writer.add_summary(
                summary=summary,
                global_step=global_step_value
            )
            if global_step_value % 500 == 0 and i != 0 and save_model_path is not None:
                # 保存模型
                import os
                save_path = os.path.join(save_model_path, str(global_step_value))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_path += '/'
                saver.save(sess, save_path)
            if i % 100 == 0:
                validation_images, validation_labels = val_data_set.get_next_batch(sub_Config.BATCH_SIZE, sub_Config.BATCH_DISTRIBUTION)
                feed_dict = {}
                for index, _ in enumerate(sub_Config.SIZES):
                    feed_dict[xs[index]] = validation_images[index]
                feed_dict[y_] = validation_labels
                validation_accuracy, validation_loss, summary, logits = sess.run(
                    [accuracy_tensor, loss_, merge_op, y],
                    feed_dict=feed_dict
                )
                calculate_acc_error(
                    logits=np.argmax(logits, 1),
                    label=validation_labels,
                    show=True
                )
                binary_acc = acc_binary_acc(
                    logits=np.argmax(logits, 1),
                    label=validation_labels,
                )
                val_writer.add_summary(summary, global_step_value)
                print 'step is %d,training loss value is %g,  accuracy is %g ' \
                      'validation loss value is %g, accuracy is %g, binary_acc is %g' % \
                      (global_step_value, loss_value, accuracy_value, validation_loss, validation_accuracy, binary_acc)
        writer.close()
        val_writer.close()
if __name__ == '__main__':
    phase_name = 'ART'
    state = ''
    traindatapath = '/home/give/Documents/dataset/MedicalImage/MedicalImage/ROIMulti/train'
    valdatapath = '/home/give/Documents/dataset/MedicalImage/MedicalImage/ROIMulti/val'
    val_dataset = ValDataSet(new_sizes=sub_Config.SIZES,
                             phase=phase_name,
                             category_number=sub_Config.OUTPUT_NODE,
                             shuffle=False,
                             data_path=valdatapath)
    train_dataset = ValDataSet(new_sizes=sub_Config.SIZES,
                               phase=phase_name,
                               category_number=sub_Config.OUTPUT_NODE,
                               shuffle=False,
                               data_path=traindatapath
                               )
    train(
        train_dataset,
        val_dataset,
        load_model_path='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/ResNetMultiScale/models/fine_tuning/5/12500/',
        save_model_path='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/ResNetMultiScale/models/fine_tuning/5/'
    )