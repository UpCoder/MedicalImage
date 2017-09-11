# -*- coding: utf-8 -*-
# 使用ｐａｔｃｈ训练好的模型，来对ＲＯＩ进行微调
from inference import inference
import tensorflow as tf
from Config import Config as sub_Config
from Tools import changed_shape, calculate_acc_error, acc_binary_acc
import numpy as np
from Patch.ValDataMultiPhase import ValDataSetMultiPhase
from Patch.Config import Config as patch_config
from Net.tools import save_weights, load


def train(train_data_set, val_data_set, phase_names, load_model_path, save_model_path):
    x = tf.placeholder(
        tf.float32,
        shape=[
            None,
            sub_Config.IMAGE_W,
            sub_Config.IMAGE_H,
            sub_Config.IMAGE_CHANNEL * len(phase_names)
        ],
        name='input_x'
    )
    if sub_Config.NEED_MUL:
        tf.summary.image(
            'input_x',
            x * 120,
            max_outputs=5
        )
    else:
        tf.summary.image(
            'input_x',
            x
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
    variable_average = tf.train.ExponentialMovingAverage(
        sub_Config.MOVING_AVERAGE_DECAY,
        global_step
    )
    vaeriable_average_op = variable_average.apply(tf.trainable_variables())
    regularizer = tf.contrib.layers.l2_regularizer(sub_Config.REGULARIZTION_RATE)
    y = inference(x, regularizer=regularizer, phase_names=phase_names)
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
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate=sub_Config.LEARNING_RATE
    ).minimize(
        loss=loss,
        global_step=global_step
    )
    with tf.control_dependencies([train_step, vaeriable_average_op]):
        train_op = tf.no_op(name='train')

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
        # sess.run(tf.global_variables_initializer())

        if load_model_path:
            # load(load_model_path, sess)
            # with tf.variable_scope('conv1_1', reuse=True):
            #     weights1 = tf.get_variable('weights')
            #     print weights1.eval(sess)
            saver.restore(sess, load_model_path)
        else:
            sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./log/5/train', tf.get_default_graph())
        val_writer = tf.summary.FileWriter('./log/5/val', tf.get_default_graph())
        for i in range(sub_Config.ITERATOE_NUMBER):
            images, labels = train_data_set.images, train_data_set.labels
            # images = changed_shape(images, [
            #         len(images),
            #         sub_Config.IMAGE_W,
            #         sub_Config.IMAGE_W,
            #         sub_Config.IMAGE_CHANNEL
            #     ])
            if i == 0:
                from PIL import Image
                image = Image.fromarray(np.asarray(images[0, :, :, 0], np.uint8))
                image.show()
            # labels[labels == 1] = 0
            # labels[labels == 3] = 0
            # labels[labels == 4] = 1
            # labels[labels == 2] = 1
            _, loss_value, accuracy_value, summary, global_step_value = sess.run(
                [train_op, loss, accuracy_tensor, merge_op, global_step],
                feed_dict={
                    x: images,
                    y_: labels
                }
            )
            writer.add_summary(
                summary=summary,
                global_step=global_step_value
            )
            if i % 500 == 0 and i != 0 and save_model_path is not None:
                # 保存模型
                saver.save(sess, save_model_path)
            if i % 100 == 0:
                validation_images, validation_labels = val_data_set.images, val_data_set.labels
                # validation_images = changed_shape(
                #     validation_images,
                #     [
                #         len(validation_images),
                #         sub_Config.IMAGE_W,
                #         sub_Config.IMAGE_W,
                #         1
                #     ]
                # )
                # validation_labels[validation_labels == 1] = 0
                # validation_labels[validation_labels == 3] = 0
                # validation_labels[validation_labels == 4] = 1
                # validation_labels[validation_labels == 2] = 1
                validation_accuracy, summary, logits = sess.run(
                    [accuracy_tensor, merge_op, y],
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
                binary_acc = acc_binary_acc(
                    logits=np.argmax(logits, 1),
                    label=validation_labels,
                )
                val_writer.add_summary(summary, global_step_value)
                print 'step is %d,training loss value is %g,  accuracy is %g ' \
                      'validation loss value is, accuracy is %g, binary_acc is %g' % \
                      (global_step_value, loss_value, accuracy_value, validation_accuracy, binary_acc)
        writer.close()
        val_writer.close()
if __name__ == '__main__':
    phase_names = ['NC', 'ART', 'PV']
    # state = '_Expand'
    state = ''
    val_dataset = ValDataSetMultiPhase(new_size=[sub_Config.IMAGE_W, sub_Config.IMAGE_H],
                                       phases=phase_names,
                                       shuffle=False,
                                       category_number=sub_Config.OUTPUT_NODE,
                                       data_path='/home/give/Documents/dataset/MedicalImage/MedicalImage/ROI' + state + '/val')
    print 'val label is '
    # print val_dataset.labels
    train_dataset = ValDataSetMultiPhase(new_size=[sub_Config.IMAGE_W, sub_Config.IMAGE_H],
                                         phases=phase_names,
                                         shuffle=False,
                                         category_number=sub_Config.OUTPUT_NODE,
                                         data_path='/home/give/Documents/dataset/MedicalImage/MedicalImage/ROI_Augmented/train')
    # print np.shape(train_dataset.labels)
    train(
        train_dataset,
        val_dataset,
        # load_model_path='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/model_finetuing/2/art/',
        phase_names,
        load_model_path=None,
        save_model_path='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/model_finetuing/5/'
    )