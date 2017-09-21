# -*- coding=utf-8 -*-
from resnet import *
import tensorflow as tf
import sys
from Config import Config as net_Config
from Tools import changed_shape

MOMENTUM = 0.99

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('save_model_path', './models', 'the saving path of the model')
tf.app.flags.DEFINE_string('log_dir', './log/train',
                           """The Summury output directory""")
tf.app.flags.DEFINE_string('log_val_dir', './log/val',
                           """The Summury output directory""")
tf.app.flags.DEFINE_float('learning_rate', 1e-9, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 32, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 500000, "max steps")
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')

'''
    计算准确率
'''
def top_k_error(predictions, labels, k):
    batch_size = float(net_Config.BATCH_SIZE) #tf.shape(predictions)[0]
    print tf.argmax(predictions, axis=1)
    print tf.cast(labels, tf.int64)
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, tf.cast(labels, tf.int64), k=k))
    num_correct = tf.reduce_sum(in_top1)
    return num_correct / batch_size


def train(train_generator, val_generator, logits, images_tensor, labeles):
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    loss_ = loss(logits, tf.cast(labeles, tf.int64))
    predictions = tf.nn.softmax(logits)
    top1_error = top_k_error(predictions, labeles, k=1)


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
        tf.summary.image('images', images)

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
    val_summary_writer = tf.summary.FileWriter(FLAGS.log_val_dir, sess.graph)
    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(FLAGS.save_model_path)
        if not latest:
            print "No checkpoint to continue from in", FLAGS.train_dir
            sys.exit(1)
        print "resume", latest
        saver.restore(sess, latest)

    for x in xrange(FLAGS.max_steps + 1):
        start_time = time.time()

        step = sess.run(global_step)
        # train_images, train_labels = train_generator.get_next_batch(net_Config.BATCH_SIZE, net_Config.BATCH_DISTRIBUTION)
        train_images, train_labels = train_generator.get_next_batch(None, None)
        train_images = changed_shape(
            train_images,
            [
                len(train_images),
                net_Config.IMAGE_W,
                net_Config.IMAGE_W,
                1
            ]
        )
        write_summary = step % 100 and step > 1
        i = [train_op, loss_]
        i.append(summary_op)
        i.append(labeles)
        o = sess.run(i, {
            images_tensor: train_images,
            labeles: train_labels
        })
        loss_value = o[1]

        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 5 == 0:
            top1_error_value = sess.run(top1_error, feed_dict={
                images_tensor: train_images,
                labeles: train_labels
            })
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('step %d loss = %.2f, accuracy = %g (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (step, loss_value, top1_error_value, examples_per_sec, duration))
        if write_summary:
            summary_str = o[2]
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step > 1 and step % 100 == 0:
            checkpoint_path = os.path.join(FLAGS.save_model_path, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)

        # Run validation periodically
        if step > 1 and step % 100 == 0:
            val_images, val_labels = val_generator.get_next_batch(net_Config.BATCH_SIZE, net_Config.BATCH_DISTRIBUTION)
            val_images = changed_shape(
                val_images,
                [
                    len(val_images),
                    net_Config.IMAGE_W,
                    net_Config.IMAGE_W,
                    1
                ]
            )
            _, top1_error_value, summary_value = sess.run([val_op, top1_error, summary_op], {
                images_tensor: val_images,
                labeles: val_labels
            })
            print('Validation accuracy %.2f' % top1_error_value)
            val_summary_writer.add_summary(summary_value, step)