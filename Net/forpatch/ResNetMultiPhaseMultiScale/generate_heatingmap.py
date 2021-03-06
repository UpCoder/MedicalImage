# -*- coding=utf-8 -*-
import os
from glob import glob
from Tools import read_mhd_image, get_boundingbox,convert2depthlaster, calculate_acc_error
import math
import numpy as np
import tensorflow as tf
from PIL import Image
from Config import Config as net_config
from resnet import inference_small
from Net.forpatch.ResNetMultiPhaseExpand.resnet_train_DIY import DataSet
phasenames=['NC', 'ART', 'PV']
mhd_adjust = False


model_path = '/home/give/PycharmProjects/MedicalImage/Net/forpatch/ResNetMultiPhaseMultiScale/model'
divided_liver = False
global_step = tf.get_variable('global_step', [],
                              initializer=tf.constant_initializer(0),
                              trainable=False)
is_training_tensor = tf.placeholder(dtype=tf.bool, shape=[])
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
batch_size_tensor = tf.placeholder(
    tf.int32,
    []
)
logits = inference_small(
    roi_images,
    expand_roi_images,
    phase_names=['NC', 'ART', 'PV'],
    num_classes=4,
    is_training=is_training_tensor,
    batch_size=batch_size_tensor
)
predictions = tf.nn.softmax(logits)
saver = tf.train.Saver(tf.all_variables())
print predictions

predicted_label_tensor = tf.argmax(predictions, axis=1)
print predicted_label_tensor
init = tf.initialize_all_variables()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
sess.run(init)
tf.train.start_queue_runners(sess=sess)
latest = tf.train.latest_checkpoint(model_path)
if not latest:
    print "No checkpoint to continue from in", model_path
    sys.exit(1)
print "resume", latest
saver.restore(sess, latest)
step = sess.run(global_step)
print 'step is ', step


def resize_images(images, size, rescale=True):
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
        # data augment
        random_int = np.random.randint(0, 4)
        img = img.rotate(random_int * 90)
        random_int = np.random.randint(0, 2)
        if random_int == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        random_int = np.random.randint(0, 2)
        if random_int == 1:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        img = img.resize([size, size])
        if rescale:
            res[i, :, :, :] = np.asarray(img, np.float32) / 255.0
            res[i, :, :, :] = res[i, :, :, :] - 0.5
            res[i, :, :, :] = res[i, :, :, :] * 2.0
        else:
            res[i, :, :, :] = np.asarray(img, np.float32)
    return res

def extract_patch(dir_name, suffix_name, patch_size, patch_step=1):
    '''
    提取指定类型病灶的ｐａｔｃｈ
    :param patch_size: 提取ｐａｔｃｈ的大小
    :param dir_name: 目前所有病例的存储路径
    :param suffix_name: 指定的病灶类型的后缀，比如说cyst 就是０
    :param patch_step: 提取ｐａｔｃｈ的步长
    :return: patch_arr (图像的个数, patch的个数)
    '''
    count = 0
    names = os.listdir(dir_name)
    patches_arr = []
    paths = []
    return_mhd_images = []
    for name in names:
        if name.endswith(suffix_name):
            # 只提取指定类型病灶的ｐａｔｃｈ
            mask_images = []
            mhd_images = []
            paths.append(os.path.join(dir_name, name))
            for phasename in phasenames:
                image_path = glob(os.path.join(dir_name, name, phasename + '_Image*.mhd'))[0]
                mask_path = os.path.join(dir_name, name, phasename + '_Registration.mhd')
                mhd_image = read_mhd_image(image_path, rejust=mhd_adjust)
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
            return_mhd_images.append(mhd_images)
            # show_image(mhd_images)
            count += 1
            [width, height, depth] = list(np.shape(mhd_images))
            patch_count = 1
            # if width * height >= 400:
            #     patch_step = int(math.sqrt(width * height / 100))
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
            patches_arr.append(patches)
            if patch_count == 1:
                continue
    print len(patches_arr)
    return patches_arr, paths, return_mhd_images

def generate_heatingmaps(data_dir, target_label, patch_size, save_dir):
    patches_arr, paths, mhd_images = extract_patch(
        data_dir,
        str(target_label),
        patch_size
    )
    from load_liver_density import load_raw_liver_density
    liver_density = load_raw_liver_density()
    for index, patches in enumerate(patches_arr):
        path = paths[index]
        basename = os.path.basename(path)
        predicted_labels = []
        start_index = 0
        while True:
            end_index = start_index + net_config.BATCH_SIZE
            if end_index > len(patches):
                #　restart = end_index - len(patches)
                end_index = len(patches)
            cur_patches = patches[start_index: end_index]
            # expand_patches = patches[start_index: end_index]
            expand_patches = [mhd_images[index]] * len(cur_patches) # 使用完整的ＲＯＩ作为ｅｘｐａｎｄ　的ｐａｔｃｈ
            roi_images_values = resize_images(cur_patches, net_config.ROI_SIZE_W, rescale=(not divided_liver))
            expand_roi_images_values = resize_images(expand_patches, net_config.EXPAND_SIZE_W, rescale=(not divided_liver))
            cur_liver_densitys = [liver_density[os.path.basename(path)]] * len(cur_patches)
            if divided_liver:
                for i in range(len(roi_images_values)):
                    for j in range(3):
                        roi_images_values[i, :, :, j] = (1.0 * roi_images_values[i, :, :, j]) / (1.0 * cur_liver_densitys[i][j])
                        expand_roi_images_values[i, :, :, j] = (1.0 * expand_roi_images_values[i, :, :, j]) / (
                            1.0 * cur_liver_densitys[i][j])
            predicted_label_value = sess.run(predicted_label_tensor, feed_dict={
                roi_images: roi_images_values,
                expand_roi_images: expand_roi_images_values,
                batch_size_tensor: len(roi_images_values),
                is_training_tensor: False
            })
            predicted_labels.extend(predicted_label_value)
            start_index = end_index
            if start_index == len(patches):
                break
        if len(predicted_labels) == 0:
            continue
        heatingmap_size = int(math.sqrt(len(predicted_labels)))
        heatingmap_image = np.zeros(
            [
                heatingmap_size,
                heatingmap_size,
                3
            ],
            np.uint8
        )
        for i in range(heatingmap_size):
            for j in range(heatingmap_size):
                heatingmap_image[i, j] = net_config.color_maping[predicted_labels[i * heatingmap_size + j]]
        print index, np.shape(heatingmap_image), len(predicted_labels)
        img = Image.fromarray(np.asarray(heatingmap_image))
        img.save(os.path.join(save_dir, str(target_label), basename+'.jpg'))


if __name__ == '__main__':
    # predicted_dir = '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/3phases/train/0'
    # names = os.listdir(predicted_dir)
    # pathes = [os.path.join(predicted_dir, name) for name in names]
    # patches = [np.array(Image.open(path)) for path in pathes]
    # generate_label(patches)

    # patches_arr = extract_patch(
    #     '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/train',
    #     '0',
    #     9
    # )
    for subclass in ['train', 'val']:
        for type in [3, 2, 1, 0]:
            generate_heatingmaps(
                '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/' + subclass,
                type,
                9,
                '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/heatingmap/' + subclass
            )