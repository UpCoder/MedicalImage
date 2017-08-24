# -*- coding: utf-8 -*-
import numpy as np
import pydicom
import os
import SimpleITK as itk
from PIL import Image, ImageDraw
import Queue
import gc
import copy
from Config import Config
import glob


# 调整窗宽 窗位
def rejust_pixel_value(image):
    image = np.array(image)
    ww = np.float64(250)
    wc = np.float64(55)
    ww = max(1, ww)
    lut_min = 0
    lut_max = 255
    lut_range = np.float64(lut_max) - lut_min

    minval = wc - ww / 2.0
    maxval = wc + ww / 2.0
    image[image < minval] = minval
    image[image > maxval] = maxval
    to_scale = (minval <= image) & (image <= maxval)
    image[to_scale] = ((image[to_scale] - minval) / (ww * 1.0)) * lut_range + lut_min
    return image


# 读取单个DICOM文件
def read_file(file_name):
    header = pydicom.read_file(file_name)
    image = header.pixel_array
    image = header.RescaleSlope * image + header.RescaleIntercept
    return image


# 读取DICOM文件序列
def read_dicom_series(dir_name):
    print 'read dicom ', dir_name
    files = list(os.listdir(dir_name))
    files.sort()
    res = []
    for file in files:
        if file.endswith('DCM'):
            cur_file = os.path.join(dir_name, file)
            res.append(read_file(cur_file))
    return res


# 读取mhd文件
def read_mhd_image(file_path):
    header = itk.ReadImage(file_path)
    image = itk.GetArrayFromImage(header)
    return np.array(image)


# 保存mhd文件
def save_mhd_image(image, file_name):
    print 'image type is ', type(image)
    header = itk.GetImageFromArray(image)
    itk.WriteImage(header, file_name)


# 将灰度图像转化为RGB通道
def conver_image_RGB(gray_image):
    shape = list(np.shape(gray_image))
    image_arr_rgb = np.zeros(shape=[shape[0], shape[1], 3])
    image_arr_rgb[:, :, 0] = gray_image
    image_arr_rgb[:, :, 1] = gray_image
    image_arr_rgb[:, :, 2] = gray_image
    return image_arr_rgb


# 将一个矩阵保存为图片
def save_image(image_arr, save_path):
    if len(np.shape(image_arr)) == 2:
        image_arr = conver_image_RGB(image_arr)
    image = Image.fromarray(np.asarray(image_arr, np.uint8))
    image.save(save_path)


# 将图像画出来，并且画出标记的病灶
def save_image_with_mask(image_arr, mask_image, save_path):
    shape = list(np.shape(image_arr))
    image_arr_rgb = np.zeros(shape=[shape[0], shape[1], 3])
    image_arr_rgb[:, :, 0] = image_arr
    image_arr_rgb[:, :, 1] = image_arr
    image_arr_rgb[:, :, 2] = image_arr
    image = Image.fromarray(np.asarray(image_arr_rgb, np.uint8))
    image_draw = ImageDraw.Draw(image)
    [ys, xs] = np.where(mask_image != 0)
    miny = np.min(ys)
    maxy = np.max(ys)
    minx = np.min(xs)
    maxx = np.max(xs)
    ROI = image_arr_rgb[miny-1:maxy+1, minx-1:maxx+1, :]
    ROI_Image = Image.fromarray(np.asarray(ROI, np.uint8))

    for index, y in enumerate(ys):
        image_draw.point([xs[index], y], fill=(255, 0, 0))
    image.save(save_path)
    ROI_Image.save(os.path.join(os.path.dirname(save_path), os.path.basename(save_path).split('.')[0]+'_ROI.jpg'))
    del image, ROI_Image
    gc.collect()
# 获取单位方向的坐标，
# 比如dim=2，则返回的数组就是[[-1, -1],...[1, 1]]
def get_direction_index(dim=3, cur_dir=[]):
    res = []
    for i in range(-1, 2):
        cur_dir.append(i)
        if dim != 1:
            res.extend(get_direction_index(dim-1, cur_dir))
            cur_dir.pop()
        else:
            res.append(copy.copy(cur_dir))
            cur_dir.pop()
    return res


# 验证数组arr1的值是否在arr_top 和 arr_dowm之间
# arr_top[i] > arr1[i] >= arr_down[i]
def value_valid(arr1, arr_top, arr_down):
    for index, item in enumerate(arr1):
        if arr_down[index] <= item < arr_top[index]:
            continue
        else:
            return False
    return True


# 将mask文件中的多个病灶拆分出来
def split_mask_image(total_mask_image_path, save_paths):
    directions = get_direction_index(dim=3)

    def find_connected_components(position, mask_image, flag):
        queue = Queue.Queue()
        points = []
        queue.put(position)
        while not queue.empty():
            cur_position = queue.get()
            points.append(cur_position)
            for direction in directions:
                new_z = cur_position[0] + direction[0]
                new_y = cur_position[1] + direction[1]
                new_x = cur_position[2] + direction[2]
                if value_valid([new_z, new_y, new_x], np.shape(mask_image), [0, 0, 0])\
                        and flag[new_z, new_y, new_x] == 1 \
                        and mask_image[new_z, new_y, new_x] != 0:
                    queue.put([new_z, new_y, new_x])
                    flag[new_z, new_y, new_x] = 0
        return points
    mask_image = read_mhd_image(total_mask_image_path)
    mask_image = np.array(mask_image)
    [z, y, x] = np.shape(mask_image)
    flag = np.ones(
        shape=[
            z, y, x
        ]
    )
    flag[np.where(mask_image == 0)] = 0
    index = 0
    for o in range(z):
        for n in range(y):
            for m in range(x):
                if mask_image[o, n, m] != 0 and flag[o, n, m] == 1:
                    flag[o, n, m] = 0
                    points = find_connected_components([o, n, m], mask_image, flag)
                    new_mask = np.zeros(
                        shape=[z, y, x]
                    )
                    for point in points:
                        new_mask[point[0], point[1], point[2]] = 1
                    save_mhd_image(new_mask, save_paths[index])
                    index += 1
                    print len(points)


# 根据Srrid判断所属的类别
def get_lesion_type_by_srrid(srrid):
    for key in Config.LESION_TYPE_RANGE.keys():
        for cur_range in Config.LESION_TYPE_RANGE[key]:
            if srrid in cur_range:
                return key
    return None


# 根据根目录读取NC，ART、PV三个phase的数据
def get_diff_phases_images(dir_path):
    images = {}
    for phase in Config.QIXIANGS:
        mhd_path = os.path.join(dir_path, phase, phase+'.mhd')
        print os.path.join(dir_path, phase)
        if os.path.exists(mhd_path):
            images[phase] = np.array(read_mhd_image(mhd_path))
        else:
            images[phase] = np.array(read_dicom_series(os.path.join(dir_path, phase)))
        if Config.ADJUST_WW_WC:
            images[phase] = rejust_pixel_value(images[phase])
    return images


# 根据目录读取LiverMask 和 所有的TumorMask
def get_total_masks(dir_path, dirs=['LiverMask', 'TumorMask']):
    tumors = {}
    if 'LiverMask' in dirs:
        liver_mask = {}
        cur_dir_path = os.path.join(dir_path, 'LiverMask')
        files = os.listdir(cur_dir_path)
        for cur_file in files:
            if not cur_file.endswith('.mhd'):
                continue
            phase_name = cur_file[cur_file.find('_', cur_file.find('_')+1) + 1: cur_file.find('.mhd')]
            liver_mask[phase_name] = read_mhd_image(os.path.join(cur_dir_path, cur_file))
        tumors['LiverMask'] = liver_mask
    if 'TumorMask' in dirs:
        tumors_mask = []
        cur_dir_path = os.path.join(dir_path, 'TumorMask')
        files = os.listdir(cur_dir_path)
        tumors_num = len(files) / 6
        for i in range(tumors_num):
            cur_tumor_mask = {}
            for phase in Config.QIXIANGS:
                mask_file_path = glob.glob(os.path.join(cur_dir_path, '*_' + phase + '_' + str(i+1) + '.mhd'))[0]
                cur_tumor_mask[phase] = read_mhd_image(mask_file_path)
            tumors_mask.append(cur_tumor_mask)
        tumors['TumorMask'] = tumors_mask
    return tumors


# 获取label的分布
def get_distribution_label(labels):
    min_value = np.min(labels)
    max_value = np.max(labels)
    my_dict = {}
    for label in labels:
        if label in my_dict:
            my_dict[label] += 1
        else:
            my_dict[label] = 0
    return my_dict


#　将数据打乱
def shuffle_image_label(images, labels):
    images = np.array(images)
    labels = np.array(labels)
    random_index = range(len(images))
    np.random.shuffle(random_index)
    images = images[random_index]
    labels = labels[random_index]
    return images, labels


# 将数据按照指定的方式排序
def changed_shape(image, shape):
    new_image = np.zeros(
        shape=shape
    )
    batch_size = shape[0]
    for z in range(batch_size):
        for phase in range(3):
            new_image[z, :, :, phase] = image[z, phase]
    del image
    gc.collect()
    return new_image


# 将一幅图像的mask外部全部标记为０
def mark_outer_zero(image, mask_image):
    def is_in(x, y, mask):
        sum1 = np.sum(mask[0:x, y])
        sum2 = np.sum(mask[x:, y])
        sum3 = np.sum(mask[x, 0:y])
        sum4 = np.sum(mask[x, y:])
        if sum1 != 0 and sum2 != 0 and sum3 != 0 and sum4 != 0:
            return True
        return False
    def fill_region(mask, x, y):
        queue = Queue.Queue()
        queue.put([x, y])
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        while not queue.empty():
            point = queue.get()
            for direction in directions:
                new_x = point[0] + direction[0]
                new_y = point[1] + direction[1]
                if value_valid([new_x, new_y], np.shape(mask), [0, 0]) and mask[new_x, new_y] == 0:
                    mask[new_x, new_y] = 1
                    queue.put([new_x, new_y])
        return mask
    [w, h] = np.shape(image)
    mask_image_copy = mask_image.copy()
    for i in range(w):
        # find = False
        for j in range(h):
            if mask_image_copy[i, j] == 0:
                if is_in(i, j, mask_image_copy):
                    print i, j
                    mask_image_copy[i, j] = 1
                    mask_image_copy = fill_region(mask_image_copy, i, j)
                    # find = True
                    image[np.where(mask_image_copy == 0)] = 0
                    return image, mask_image_copy
    print 'Error'
    return image, mask_image_copy


# 显示一幅图像
def show_image(image_arr):
    image = Image.fromarray(image_arr)
    image.show()


# 计算Ａｃｃｕｒａｃｙ，并且返回每一类最大错了多少个
def calculate_acc_error(logits, label, show=True):
    error_dict = {}
    error_dict_record = {}
    error_count = 0
    for index, logit in enumerate(logits):
        if logit != label[index]:
            error_count += 1
            if label[index] in error_dict.keys():
                error_dict[label[index]] += 1   # 该类别分类错误的个数加１
                error_dict_record[label[index]].append(logit)   # 记录错误的结果
            else:
                error_dict[label[index]] = 1
                error_dict_record[label[index]] = [logit]
    acc = (1.0 * error_count) / (1.0 * len(label))
    if show:
        for key in error_dict.keys():
            print 'label is %d, error number is %d' % (key, error_dict[key])
            print 'error record　is ', error_dict_record[key]
    return error_dict, error_dict_record, acc
if __name__ == '__main__':
    # dicom_path = 'E:\\Resource\\DataSet\\MedicalImage\\METS\METS\\177-2977086\PV'
    # read_dicom_series(dicom_path)
    series_path = '/home/give/Documents/dataset/MedicalImage/MedicalImage/CYST/000-2945085/ART'
    mask_image_path = '/home/give/Documents/dataset/MedicalImage/MedicalImage/CYST/000-2945085/TumorMask/TumorMask_Srr000_ART_1.mhd'
    mask_image = read_mhd_image(mask_image_path)
    images = read_dicom_series(series_path)
    images = np.array(images)
    print np.shape(images)
    for index, image in enumerate(images):
        images[index, :, :] = rejust_pixel_value(images[index, :, :])
    img = Image.fromarray(images[10, :, :])
    img.show()

    mask_img = Image.fromarray(mask_image[60, :, :] * 255)
    mask_img.show()

    zero_img, zero_mask = mark_outer_zero(images[10, :, :], mask_image[60, :, :])
    zero_img = Image.fromarray(zero_img)
    zero_mask = Image.fromarray(zero_mask * 255)
    zero_img.show()
    zero_mask.show()
