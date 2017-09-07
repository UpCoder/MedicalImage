# -*- coding: utf-8 -*-
from inference import inference
import tensorflow as tf
from Config import Config as sub_Config
from Slice.MaxSlice.MaxSlice_Resize import MaxSlice_Resize
from Tools import changed_shape, calculate_acc_error
import numpy as np
from param_shared import inference as inference_parllel


def train(dataset, load_model=False):
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
    tf.summary.image(
        'input_x1',
        x1,
        max_outputs=10
    )
    tf.summary.image(
        'input_x2',
        x2,
        max_outputs=10
    )
    tf.summary.image(
        'input_x2',
        x3,
        max_outputs=10
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
    regularizer = tf.contrib.layers.l2_regularizer(sub_Config.REGULARIZTION_RATE)
    y = inference_parllel([x1, x2, x3], regularizer)
    tf.summary.histogram(
        'logits',
        tf.argmax(y, 1)
    )
    variable_averages = tf.train.ExponentialMovingAverage(
        sub_Config.MOVING_AVERAGE_DECAY, global_step
    )
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables()
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
    with tf.control_dependencies([train_step, variable_averages_op]):
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
            images1 = images[:, 0, :]
            images2 = images[:, 1, :]
            images3 = images[:, 2, :]
            images1 = changed_shape(images1, [
                sub_Config.BATCH_SIZE,
                sub_Config.sizes[0][0],
                sub_Config.sizes[0][1],
                sub_Config.sizes[0][2]
            ])
            images2 = changed_shape(images2, [
                sub_Config.BATCH_SIZE,
                sub_Config.sizes[1][0],
                sub_Config.sizes[1][1],
                sub_Config.sizes[1][2]
            ])
            images3 = changed_shape(images3, [
                sub_Config.BATCH_SIZE,
                sub_Config.sizes[2][0],
                sub_Config.sizes[2][1],
                sub_Config.sizes[2][2]
            ])

            if i == 0:
                from PIL import Image
                image = Image.fromarray(np.asarray(images3[0, :, :, 0], np.uint8))
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
            _, loss_value, accuracy_value, summary, global_step_value = sess.run(
                [train_op, loss, accuracy_tensor, merge_op, global_step],
                feed_dict={
                    x1: images1,
                    x2: images2,
                    x3: images3,
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
                print 'step is %d,training loss value is %g,  accuracy is %g ' \
                      'validation loss value is %g, accuracy is %g' % \
                      (global_step_value, loss_value, accuracy_value, validation_loss, validation_accuracy)
                val_writer.add_summary(summary, i)
        writer.close()
        val_writer.close()
if __name__ == '__main__':
    dataset = MaxSlice_Resize(sub_Config)
    # mnist = input_data.read_data_sets("../data", one_hot=True)
    # train(mnist)