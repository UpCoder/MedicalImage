# -*- coding=utf-8 -*-
import os
from glob import glob
from Tools import read_mhd_image, get_boundingbox,convert2depthlaster
import math
import numpy as np
import tensorflow as tf
from PIL import Image
from Config import Config as net_config
from resnet import inference_small
phasenames=['NC', 'ART', 'PV']

def resize_images(images, size):
    shape = list(np.shape(images))
    res = np.zeros(
        [
            shape[0],
            size,
            size,
            3
        ],
        np.float32
    )
    for i in range(shape[0]):
        img = Image.fromarray(np.asarray(images[i], np.uint8))
        img = img.resize([size, size])
        res[i, :, :, :] = np.asarray(img, np.float32) / 255.0
        res[i, :, :, :] = res[i, :, :, :] - 0.5
        res[i, :, :, :] = res[i, :, :, :] * 2.0
    return res

def generate_label(patches, model_path='/home/give/PycharmProjects/MedicalImage/Net/forpatch/ResNetMultiPhaseExpand/models'):
    if len(patches) > net_config.BATCH_SIZE:
        patches = patches[:100]
    roi_image_values = resize_images(patches, net_config.ROI_SIZE_W)
    expand_roi_images_values = resize_images(patches, net_config.EXPAND_SIZE_W)

    roi_images = tf.placeholder(
        shape=[
            None,
            net_config.ROI_SIZE_W,
            net_config.ROI_SIZE_H,
            net_config.IMAGE_CHANNEL
        ],
        dtype=np.float32,
        name='roi_input'
    )
    expand_roi_images = tf.placeholder(
        shape=[
            None,
            net_config.EXPAND_SIZE_W,
            net_config.EXPAND_SIZE_H,
            net_config.IMAGE_CHANNEL
        ],
        dtype=np.float32,
        name='expand_roi_input'
    )
    logits = inference_small(
        roi_images,
        expand_roi_images,
        phase_names=['NC', 'ART', 'PV'],
        num_classes=4,
        is_training=True,
    )
    predictions = tf.nn.softmax(logits)
    saver = tf.train.Saver(tf.all_variables())
    print predictions

    predicted_label_tensor = tf.argmax(predictions, axis=1)
    print predicted_label_tensor
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        latest = tf.train.latest_checkpoint(model_path)
        if not latest:
            print "No checkpoint to continue from in", model_path
            sys.exit(1)
        print "resume", latest
        saver.restore(sess, latest)

        predicted_label_value = sess.run(predicted_label_tensor, feed_dict={
            roi_images: roi_image_values,
            expand_roi_images: expand_roi_images_values
        })
        print predicted_label_value


if __name__ == '__main__':
    predicted_dir = '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/3phases/train/0'
    names = os.listdir(predicted_dir)
    pathes = [os.path.join(predicted_dir, name) for name in names]
    patches = [np.array(Image.open(path)) for path in pathes]
    generate_label(patches)

def extract_patch(dir_name, suffix_name, save_dir, patch_size, patch_step=1):
    '''
    提取指定类型病灶的ｐａｔｃｈ
    :param patch_size: 提取ｐａｔｃｈ的大小
    :param dir_name: 目前所有病例的存储路径
    :param suffix_name: 指定的病灶类型的后缀，比如说cyst 就是０
    :param save_dir:　提取得到的ｐａｔｃｈ的存储路径
    :param patch_step: 提取ｐａｔｃｈ的步长
    :return: None
    '''
    count = 0
    names = os.listdir(dir_name)
    for name in names:
        if name.endswith(suffix_name):
            # 只提取指定类型病灶的ｐａｔｃｈ
            mask_images = []
            mhd_images = []
            for phasename in phasenames:
                image_path = glob(os.path.join(dir_name, name, phasename + '_Image*.mhd'))[0]
                mask_path = os.path.join(dir_name, name, phasename + '_Registration.mhd')
                mhd_image = read_mhd_image(image_path, rejust=True)
                mhd_image = np.squeeze(mhd_image)
                # show_image(mhd_image)
                mask_image = read_mhd_image(mask_path)
                mask_image = np.squeeze(mask_image)
                [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)
                # xmin -= 15
                # xmax += 15
                # ymin -= 15
                # ymax += 15
                mask_image = mask_image[xmin: xmax, ymin: ymax]
                mhd_image = mhd_image[xmin: xmax, ymin: ymax]
                mhd_image[mask_image != 1] = 0
                mask_images.append(mask_image)
                mhd_images.append(mhd_image)
                # show_image(mhd_image)
            mask_images = convert2depthlaster(mask_images)
            mhd_images = convert2depthlaster(mhd_images)
            # show_image(mhd_images)
            count += 1
            [width, height, depth] = list(np.shape(mhd_images))
            patch_count = 1
            if width * height >= 400:
                patch_step = int(math.sqrt(width * height / 100))
            heatingmap = np.zeros([width, height, 3], dtype=np.uint8)
            patches = []
            for i in range(patch_size / 2, width - patch_size / 2, patch_step):
                for j in range(patch_size / 2, height - patch_size / 2, patch_step):
                    cur_patch = mhd_images[i - patch_size / 2:i + patch_size / 2,
                                j - patch_size / 2: j + patch_size / 2, :]
                    if (np.sum(
                            mask_images[i - patch_size / 2:i + patch_size / 2, j - patch_size / 2: j + patch_size / 2,
                            :]) / ((patch_size - 1) * (patch_size - 1) * 3)) < 0.9:
                        continue
                    patches.append(cur_patch)
                    # save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.png')
                    # patch_image = Image.fromarray(np.asarray(cur_patch, np.uint8))
                    # patch_image.save(save_path)
                    # patch_count += 1

            if patch_count == 1:
                save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.png')
                roi_image = Image.fromarray(np.asarray(mhd_images, np.uint8))
                roi_image.save(save_path)

    print count