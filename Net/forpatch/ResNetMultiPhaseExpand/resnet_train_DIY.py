# -*- coding=utf-8 -*-
from resnet import *
import tensorflow as tf
from Tools import changed_shape, calculate_acc_error, acc_binary_acc, shuffle_image_label
from glob import glob
import shutil
from Config import Config as net_config
from PIL import Image

MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('load_model_path', './models/DIY',
                           '''the model reload path''')
tf.app.flags.DEFINE_string('save_model_path', './models', 'the saving path of the model')
tf.app.flags.DEFINE_string('log_dir', './log/train',
                           """The Summury output directory""")
tf.app.flags.DEFINE_string('log_val_dir', './log/val',
                           """The Summury output directory""")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('max_steps', 10000, "max steps")
tf.app.flags.DEFINE_boolean('resume', True,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')


def top_k_error(predictions, labels, k):
    batch_size = float(FLAGS.batch_size) #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size

def calculate_accuracy(logits, labels, arg_index=1):
    """Evaluate the quality of the logits at predicting the label.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor,
    """
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.arg_max(logits, arg_index), tf.arg_max(labels, arg_index))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope+'/accuracy', accuracy)
    return accuracy
class DataSet:
    @staticmethod
    def resize_images(images, size):
        res = np.zeros(
            [
                len(images),
                size,
                size,
                3
            ],
            np.float32
        )
        for i in range(len(images)):
            img = Image.fromarray(np.asarray(images[i], np.uint8))
            img = img.resize([size, size])
            res[i, :, :, :] = np.asarray(img, np.float32) / 255.0
            res[i, :, :, :] = res[i, :, :, :] - 0.5
            res[i, :, :, :] = res[i, :, :, :] * 2.0
        return res
    @staticmethod
    def generate_paths(dir_name, state, target_labels=[0, 1, 2, 3], shuffle=True):
        '''
        返回dirname中的所有病灶图像的路径
        :param dir_name:  父文件夹的路径
        :param state: 状态，一般来说父文件夹有两个状态 train 和val
        :param target_labels: 需要文件标注的label
        :return:
        '''
        roi_paths = []
        roi_expand_paths = []
        labels = []

        cur_dir = os.path.join(dir_name, state)
        # names = os.listdir(cur_dir)
        for target_label in target_labels:
            type_dir = os.path.join(cur_dir, str(target_label))
            type_names = os.listdir(type_dir)
            roi_paths.extend([os.path.join(type_dir, name) for name in type_names])
            labels.extend([target_label] * len(type_names))
        if shuffle:
            roi_paths, labels = shuffle_image_label(roi_paths, labels)
        return roi_paths, roi_paths, labels

    def __init__(self, data_dir, state):
        self.roi_paths, self.expand_roi_path, self.labels = DataSet.generate_paths(
            data_dir,
            state
        )
        self.state = state
        self.epoch_num = 0
        self.start_index = 0

    def get_next_batch(self, batch_size):
        while True:
            cur_roi_paths = []
            cur_expand_roi_paths = []
            cur_labels = []
            end_index = self.start_index + batch_size
            if end_index > len(self.roi_paths):
                self.epoch_num += 1
                cur_roi_paths.extend(self.roi_paths[self.start_index: len(self.roi_paths)])
                cur_roi_paths.extend(self.roi_paths[:end_index - len(self.roi_paths)])

                cur_expand_roi_paths.extend(self.expand_roi_path[self.start_index: len(self.roi_paths)])
                cur_expand_roi_paths.extend(self.expand_roi_path[:end_index - len(self.roi_paths)])

                cur_labels.extend(self.labels[self.start_index: len(self.roi_paths)])
                cur_labels.extend(self.labels[:end_index - len(self.roi_paths)])
                self.start_index = end_index - len(self.roi_paths)
                print 'state: ', self.state, ' epoch: ', self.epoch_num
            else:
                cur_roi_paths.extend(self.roi_paths[self.start_index: end_index])
                cur_expand_roi_paths.extend(self.expand_roi_path[self.start_index: end_index])
                cur_labels.extend(self.labels[self.start_index: end_index])
                self.start_index = end_index
            cur_roi_images = [np.asarray(Image.open(path)) for path in cur_roi_paths]
            cur_expand_roi_images = [np.asarray(Image.open(path)) for path in cur_expand_roi_paths]
            cur_roi_images = DataSet.resize_images(cur_roi_images, net_config.ROI_SIZE_W)
            cur_expand_roi_images = DataSet.resize_images(cur_expand_roi_images, net_config.EXPAND_SIZE_W)
            # print np.shape(cur_roi_images)
            yield cur_roi_images, cur_expand_roi_images, cur_labels

def train(logits, images_tensor, expand_images_tensor, labels_tensor, save_model_path=None, step_width=100):

    train_dataset = DataSet('/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/3phases', 'train')
    val_dataset = DataSet('/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/3phases', 'val')

    train_batchdata = train_dataset.get_next_batch(net_config.BATCH_SIZE)
    val_batchdata = val_dataset.get_next_batch(net_config.BATCH_SIZE)

    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    loss_ = loss(logits, labels_tensor)
    predictions = tf.nn.softmax(logits)
    print 'predictions shape is ', predictions
    print 'label is ', labels_tensor
    top1_error = top_k_error(predictions, labels_tensor, 1)
    labels_onehot = tf.one_hot(labels_tensor, logits.get_shape().as_list()[-1])
    print 'output node is ', logits.get_shape().as_list()[-1]
    accuracy_tensor = calculate_accuracy(predictions, labels_onehot)

    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    tf.summary.scalar('loss_avg', ema.average(loss_))

    # validation stats
    ema = tf.train.ExponentialMovingAverage(0.9, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
    top1_error_avg = ema.average(top1_error)
    tf.summary.scalar('val_top1_error_avg', top1_error_avg)

    tf.summary.scalar('learning_rate', FLAGS.learning_rate)

    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, MOMENTUM)
    grads = opt.compute_gradients(loss_)
    for grad, var in grads:
        if grad is not None and not FLAGS.minimal_summaries:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    if not FLAGS.minimal_summaries:
        # Display the training images in the visualizer.
        tf.summary.image('images', images_tensor)

        for var in tf.trainable_variables():
            tf.summary.image(var.op.name, var)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver = tf.train.Saver(tf.all_variables())

    summary_op = tf.summary.merge_all()

    init = tf.initialize_all_variables()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    # val_summary_writer = tf.summary.FileWriter(FLAGS.log_val_dir, sess.graph)
    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(FLAGS.load_model_path)
        if not latest:
            print "No checkpoint to continue from in", FLAGS.train_dir
            sys.exit(1)
        print "resume", latest
        saver.restore(sess, latest)

    for x in xrange(FLAGS.max_steps + 1):
        start_time = time.time()

        step = sess.run(global_step)
        i = [train_op, loss_]

        write_summary = step % 100 and step > 1
        if write_summary:
            i.append(summary_op)
        train_roi_batch_images, train_expand_roi_batch_images, train_labels = train_batchdata.next()
        o = sess.run(i, feed_dict={
            images_tensor: train_roi_batch_images,
            expand_images_tensor: train_expand_roi_batch_images,
            labels_tensor: train_labels
        })

        loss_value = o[1]

        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if (step - 1) % step_width == 0:
            train_labels = np.array(train_labels)
            train_roi_batch_images = np.array(train_roi_batch_images)
            train_expand_roi_batch_images = np.array(train_expand_roi_batch_images)

            target_label = 0
            train_roi_batch_images = train_roi_batch_images[train_labels == target_label]
            train_expand_roi_batch_images = train_expand_roi_batch_images[train_labels == target_label]
            train_labels = train_labels[train_labels == target_label]

            top1_error_value, accuracy_value, labels_values, predictions_values = sess.run([top1_error, accuracy_tensor, labels_tensor, predictions], feed_dict={
                images_tensor: train_roi_batch_images,
                expand_images_tensor: train_expand_roi_batch_images,
                labels_tensor: train_labels
            })
            predictions_values = np.argmax(predictions_values, axis=1)
            examples_per_sec = FLAGS.batch_size / float(duration)
            # accuracy = eval_accuracy(predictions_values, labels_values)
            format_str = ('step %d, loss = %.2f, top1 error = %g, accuracy value = %g  (%.1f examples/sec; %.3f '
                          'sec/batch)')

            print(format_str % (step, loss_value, top1_error_value, accuracy_value, examples_per_sec, duration))
        if write_summary:
            summary_str = o[2]
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step > 1 and step % step_width == 0:
            checkpoint_path = os.path.join(save_model_path, 'model.ckpt')
            # saver.save(sess, checkpoint_path, global_step=global_step)
            # save_dir = os.path.join(save_model_path, str(step))
            # if not os.path.exists(save_dir):
            #     os.mkdir(save_dir)
            # filenames = glob(os.path.join(save_model_path, '*-'+str(int(step + 1))+'.*'))
            # for filename in filenames:
            #     shutil.copy(
            #         filename,
            #         os.path.join(save_dir, os.path.basename(filename))
            #     )
        # Run validation periodically
        if step > 1 and step % step_width == 0:
            val_roi_batch_images, val_expand_roi_batch_images, val_labels = val_batchdata.next()
            _, top1_error_value, summary_value, accuracy_value, labels_values, predictions_values = sess.run(
                [val_op, top1_error, summary_op, accuracy_tensor, labels_tensor, predictions],
                {
                    images_tensor: val_roi_batch_images,
                    expand_images_tensor: val_expand_roi_batch_images,
                    labels_tensor: val_labels
                })
            predictions_values = np.argmax(predictions_values, axis=1)
            # accuracy = eval_accuracy(predictions_values, labels_values)
            calculate_acc_error(
                logits=predictions_values,
                label=labels_values,
                show=True
            )
            print('Validation top1 error %.2f, accuracy value %f'
                  % (top1_error_value, accuracy_value))
            # val_summary_writer.add_summary(summary_value, step)