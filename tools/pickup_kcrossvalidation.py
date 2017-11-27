# -*- coding=utf-8 -*-
import os
from shutil import copytree
import numpy as np
def split2kcrossvalidation(data_dir, target_dir, k=3):
    '''
    将我们原始的训练数据拆分成k分，然后进行交叉验证
    :param data_dir: 原始的文件夹
    :param k: 拆分得到的份数
    :param target_dir: 目标的文件夹
    :return:
    '''
    names = np.array(os.listdir(data_dir))
    index = range(len(names))
    np.random.shuffle(index)
    names = names[index]
    pre_num = len(names) / k
    start = 0
    for i in range(k):
        target_path = os.path.join(target_dir, str(i))
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        if i != (k-1):
            end = start + pre_num
        else:

            end = len(names)
        will_copy_names = names[start: end]
        start = end
        for name in will_copy_names:
            copytree(
                os.path.join(data_dir, name),
                os.path.join(target_path, name)
                # target_path
            )
            print os.path.join(data_dir, name), '-->', target_path
def statics_distribution(cross_dir):
    '''
    展示生成的数据的分布
    :param cross_dir:
    :return:
    '''
    for kcorssname in os.listdir(cross_dir):
        distribution = {}
        for name in os.listdir(os.path.join(cross_dir, kcorssname)):
            type_id = name[-1]
            if type_id in distribution.keys():
                distribution[type_id] += 1
            else:
                distribution[type_id] = 1
        print 'kcrossname is ', kcorssname
        print 'responding distribution is ', distribution

def correct():
    from Net.forpatch.cross_validation.resnet_train_DIY import DataSet
    cross_id = 0
    cross_ids = [0, 1, 2]
    del cross_ids[cross_id]
    train_dataset = DataSet('/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/cross_validation', 'train',
                            rescale=True, divied_liver=False, expand_is_roi=True, cross_ids=cross_ids,
                            full_roi_path='/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/train')
    val_dataset = DataSet('/home/give/Documents/dataset/MedicalImage/MedicalImage/Patches/cross_validation', 'val',
                          rescale=True, divied_liver=False, expand_is_roi=True, cross_ids=[cross_id],
                          full_roi_path='/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/train')
    train_batchdata = train_dataset.get_next_batch(100)
    val_batchdata = val_dataset.get_next_batch(100)
    while True:
        # train_batchdata.next()
        val_batchdata.next()
if __name__ == '__main__':
    # split2kcrossvalidation(
    #     '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/train',
    #     '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/train_cross',
    #     k=3
    # )
    # statics_distribution('/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/train_cross')
    correct()