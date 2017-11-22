# -*- coding=utf-8 -*-
import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from glob import glob
from Tools import read_mhd_image, get_boundingbox, convert2depthlaster
import scipy.io as scio
phasenames=['NC', 'ART', 'PV']
patch_size = 4


def load_patch(patch_path):
    if patch_path.endswith('.jpg'):
        return Image.open(patch_path)
    if patch_path.endswith('.npy'):
        return np.load(patch_path)

def generate_density_feature(data_dir):
    names = os.listdir(data_dir)
    features = []
    for name in names:

        array = np.array(load_patch(os.path.join(data_dir, name))).flatten()
        if len(array) != patch_size * patch_size * 3:
            continue
        features.append(array)
    # features = [np.array(Image.open(os.path.join(data_dir, name))).flatten() for name in names]
    return np.array(features)


def generate_density_feature_multidir(data_dirs):
    features = []
    for data_dir in data_dirs:
        features.extend(
            generate_density_feature(data_dir)
        )
    return np.array(features)


def do_kmeans(fea, vocabulary_size=256):
    kmeans_obj = KMeans(n_clusters=vocabulary_size, n_jobs=8).fit(fea)
    cluster_centroid_objs = kmeans_obj.cluster_centers_
    np.save(
        './cluster_centroid_objs_'+str(vocabulary_size)+'.npy',
        cluster_centroid_objs
    )
    print np.shape(cluster_centroid_objs)


def extract_patch(dir_name, suffix_name, patch_size, patch_step=1, flatten=False):
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
    for name in names:
        if name.endswith(suffix_name):
            # 只提取指定类型病灶的ｐａｔｃｈ
            mask_images = []
            mhd_images = []
            paths.append(os.path.join(dir_name, name))
            for phasename in phasenames:
                image_path = glob(os.path.join(dir_name, name, phasename + '_Image*.mhd'))[0]
                mask_path = os.path.join(dir_name, name, phasename + '_Registration.mhd')
                mhd_image = read_mhd_image(image_path, rejust=True)
                mhd_image = np.squeeze(mhd_image)
                # show_image(mhd_image)
                mask_image = read_mhd_image(mask_path)
                mask_image = np.squeeze(mask_image)
                [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)
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
                    patch_count += 1
                    if flatten:
                        patches.append(np.array(cur_patch).flatten())
                    else:
                        patches.append(cur_patch)
            if patch_count == 1:
                continue
            patches_arr.append(patches)
    return np.array(patches_arr)

def generate_train_val_features(cluster_centroid_path):
    train_patches = []
    train_labels = []
    for i in range(4):
        cur_patches = extract_patch(
                '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/train',
                str(i),
                patch_size=(patch_size + 1),
                flatten=True
            )
        train_labels.extend([i] * len(cur_patches))
        train_patches.extend(
            cur_patches
        )
    train_patches = np.array(train_patches)
    print 'train_patches shape is ', np.shape(train_patches)

    val_patches = []
    val_labels = []
    for i in range(4):
        cur_patches = extract_patch(
                '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/val',
                str(i),
                patch_size=(patch_size + 1),
                flatten=True
            )
        val_labels.extend([i] * len(cur_patches))
        val_patches.extend(
            cur_patches
        )
    val_patches = np.array(val_patches)
    print 'val_patches shape is ', np.shape(val_patches)

    cluster_centroid_arr = np.load(cluster_centroid_path)
    train_features = []
    val_features = []
    for i in range(len(train_patches)):
        train_features.append(
            generate_patches_representer(train_patches[i], cluster_centroid_arr).squeeze()
        )
    for i in range(len(val_patches)):
        val_features.append(
            generate_patches_representer(val_patches[i], cluster_centroid_arr).squeeze()
        )
    print 'the shape of train features is ', np.shape(train_features)
    print 'the shape of val features is ', np.shape(val_features)
    scio.savemat(
        './data.mat',
        {
            'train_features': train_features,
            'train_labels': train_labels,
            'val_features': val_features,
            'val_labels': val_labels
        }
    )
def generate_patches_representer(patches, cluster_centers):
    '''
    用词典表示一组patches
    :param patches: 表示一组patch　（None, 192）
    :param cluster_centers:　(vocabulary_size, 192)
    :return: (1, vocabulary_size) 行向量　表示这幅图像
    '''
    print np.shape(patches)
    print np.shape(cluster_centers)
    shape = list(np.shape(cluster_centers))
    mat_cluster_centers = np.mat(cluster_centers)
    mat_patches = np.mat(patches)
    mat_distance = mat_patches * mat_cluster_centers.T # (None, vocabulary_size)
    represented_vector = np.zeros([1, shape[0]])
    for i in range(len(mat_distance)):
        distance_vector = np.array(mat_distance[i])
        min_index = np.argmin(distance_vector, axis=1)
        represented_vector[0, min_index] += 1
    return represented_vector
if __name__ == '__main__':
    # features = generate_density_feature_multidir(
    #     [
    #         '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/3phases_3*3/balance/val/0',
    #         '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/3phases_3*3/balance/val/1',
    #         '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/3phases_3*3/balance/val/2',
    #         '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/3phases_3*3/balance/val/3',
    #         '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/3phases_3*3/balance/val/0',
    #         '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/3phases_3*3/balance/val/1',
    #         '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/3phases_3*3/balance/val/2',
    #         '/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/3phases_3*3/balance/val/3',
    #     ]
    # )
    # print np.shape(features)
    # do_kmeans(features)

    generate_train_val_features('/home/give/PycharmProjects/MedicalImage/BoVW/cluster_centroid_objs_256.npy')