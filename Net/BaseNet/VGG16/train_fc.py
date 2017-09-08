# -*- coding: utf-8 -*-
import tensorflow as tf
from trained_vgg16 import vgg16
from Patch.PatchBase import PatchDataSet
from Patch.Config import Config as patch_config
import numpy as np
import gc
from Config import Config as net_config
from Tools import changed_shape, calculate_acc_error
from Net.tools import calculate_accuracy, calculate_loss

class train:
    def __init__(self, load_model_path, save_model_path):
        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self.BATCH_SIZE = 128
        self.sess = tf.Session()
        imgs = tf.placeholder(
            tf.float32,
            shape=[
                None,
                net_config.IMAGE_W,
                net_config.IMAGE_H,
                net_config.IMAGE_CHANNEL
            ]
        )
        self.dataset = PatchDataSet(config=patch_config, new_size=[net_config.IMAGE_W, net_config.IMAGE_H])
        self.learning_rate = 1e-3
        self.iterator_number = int(1e+5)
        # self.params_path = '/home/give/PycharmProjects/StomachCanner/classification/Net/VGG16/vgg16.npy'
        self.params_path = None
        self.vgg = vgg16(imgs, self.params_path, self.sess, skip_layers=['fc8'])
        
    def one_hot_encoding(self, labels):
        nb_classes = net_config.OUTPUT_NODE
        targets = np.array([labels]).reshape(-1)
        one_hot_targets = np.eye(nb_classes)[targets]
        return one_hot_targets

    def start_train(self):
        y_ = tf.placeholder(
            tf.float32,
            [
                None,
                net_config.OUTPUT_NODE
            ]
        )
        # avg_image = calu_average_train_set(config.TRAIN_DATA_DIR, [net_config.IMAGE_W, net_config.IMAGE_H])
        y = self.vgg.fcs_output
        global_step = tf.Variable(0, trainable=False)
        variable_averages = tf.train.ExponentialMovingAverage(
            net_config.MOVEING_AVERAGE_DECAY,
            global_step
        )
        variable_averages_op = variable_averages.apply(
            tf.trainable_variables()
        )
        loss = calculate_loss(logits=y, labels=y_)
        tf.summary.scalar(
            'loss',
            loss
        )
        train_step = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate
        ).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')
        # 计算准确率
        accuracy_tensor = calculate_accuracy(logits=y, labels=y_)
        merged = tf.summary.merge_all()
        max_accuracy = 0.0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            log_path = './log/train'
            val_log_path = './log/val'
            if self.load_model_path:
                saver.restore(sess, self.load_model_path)
            writer = tf.summary.FileWriter(log_path, tf.get_default_graph())
            val_writer = tf.summary.FileWriter(val_log_path, tf.get_default_graph())
            for i in range(self.iterator_number):
                train_images, labels = self.dataset.get_next_train_batch(net_config.TRAIN_BATCH_SIZE)
                labels = self.one_hot_encoding(labels)
                train_images = changed_shape(
                    train_images,
                    [
                        len(train_images),
                        net_config.IMAGE_W,
                        net_config.IMAGE_W,
                        1
                    ]
                )
                feed_dict = {
                    self.vgg.imgs: train_images,
                    y_: labels
                }
                _, loss_value, accuracy_value, summary, y_value, global_step_value = sess.run(
                    [train_op, loss, accuracy_tensor, merged, y, global_step],
                    feed_dict=feed_dict
                )
                if i % 500 == 0 and i != 0 and self.save_model_path is not None:
                    # 保存模型
                    print 'save model successful', self.save_model_path
                    saver.save(sess, self.save_model_path)
                writer.add_summary(summary, i)
                if (i % 40) == 0 and i != 0:
                    val_images, labels = self.dataset.get_next_val_batch(net_config.TRAIN_BATCH_SIZE)
                    val_images = changed_shape(
                        val_images,
                        [
                            len(val_images),
                            net_config.IMAGE_W,
                            net_config.IMAGE_W,
                            1
                        ]
                    )
                    labels = self.one_hot_encoding(labels)
                    feed_dict = {
                        self.vgg.imgs: val_images,
                        y_: labels
                    }
                    val_loss, val_accuracy, summary = sess.run(
                        [loss, accuracy_tensor, merged],
                        feed_dict=feed_dict
                    )
                    val_writer.add_summary(summary, i)
                    print '-'*15, 'global_step is %d, train loss value is %g, accuracy is %g, val loss is %g, val accuracy is %g' % \
                                  (global_step_value, loss_value, accuracy_value, val_loss, val_accuracy), '-'*15
                del train_images, labels
                gc.collect()
        writer.close()
        val_writer.close()

if __name__ == '__main__':
    my_train = train(
        load_model_path=None,
        save_model_path='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/VGG16/model/'
    )
    my_train.start_train()