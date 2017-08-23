from inference import inference
import tensorflow as tf
from Config import Config as sub_Config
from Slice.MaxSlice.MaxSlice_Resize import MaxSlice_Resize
from tensorflow.examples.tutorials.mnist import input_data
from Tools import changed_shape
import numpy as np


def train(dataset):
    x = tf.placeholder(
        tf.float32,
        shape=[
            sub_Config.BATCH_SIZE,
            sub_Config.IMAGE_W,
            sub_Config.IMAGE_H,
            sub_Config.IMAGE_CHANNEL
        ],
        name='input_x'
    )
    tf.summary.image(
        'input_x',
        x,
        max_outputs=10
    )
    y_ = tf.placeholder(
        tf.float32,
        shape=[
            sub_Config.BATCH_SIZE,
        ]
    )
    tf.summary.histogram(
        'label',
        y_
    )
    y = inference(x)
    tf.summary.histogram(
        'logits',
        tf.argmax(y, 1)
    )
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y,
            labels=tf.cast(y_, tf.int32)
        )
    )
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
    merge_op = tf.summary.merge_all()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(sub_Config.LOG_DIR, tf.get_default_graph())
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
            images = np.reshape(
                images[:, :, :, 2],
                [
                    sub_Config.BATCH_SIZE,
                    sub_Config.IMAGE_W,
                    sub_Config.IMAGE_W,
                    1
                ]
            )
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
            if i % 20 == 0:
                print 'step is %d, loss value is %g, accuracy is %g' % \
                      (i, loss_value, accuracy_value)
        writer.close()
if __name__ == '__main__':
    dataset = MaxSlice_Resize(sub_Config)
    # mnist = input_data.read_data_sets("../data", one_hot=True)
    # train(mnist)