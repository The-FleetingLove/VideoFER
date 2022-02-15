import os
import shutil
import pandas as pd
import numpy as np


def convert_num(str_path):
    return int(str_path.split('.')[0])


def replace_path(dir_path):
    return dir_path.replace('\\', '/')


# 源数据路径
annotation_train_path = 'D:/FER/Dataset/Aff-Wild2/annotations/VA_Set/Train_Set/'
annotation_eval_path = 'D:/FER/Dataset/Aff-Wild2/annotations/VA_Set/Validation_Set/'
img_path_sum = 'D:/FER/Dataset/Aff-Wild2/cropped_aligned/'
# 处理后数据路径
currentDir = replace_path(os.getcwd()) + '/'  # 用于返回当前工作目录
data_dir = os.path.join(currentDir, 'Aff_data/')  # 存放处理后的数据文件夹
train_img_dir = os.path.join(data_dir, 'train_img/')  # 存放训练集图片文件夹
eval_img_dir = os.path.join(data_dir, 'eval_img/')  # 存放训练集图片文件夹
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(train_img_dir):
    os.makedirs(train_img_dir)
if not os.path.exists(eval_img_dir):
    os.makedirs(eval_img_dir)
print("{}\n{}\n{}\n{}\n".format(currentDir, data_dir, train_img_dir, eval_img_dir))  # 测试文件夹目录路径
# 存放处理数据的变量
csv_train_v = []
csv_train_a = []
csv_train_img = []
csv_eval_v = []
csv_eval_a = []
csv_eval_img = []

# 处理训练集
for _, _, files in os.walk(annotation_train_path):
    sort_files = sorted(files)
    for file_name in sort_files:
        label_path = os.path.join(annotation_train_path, file_name)  # 标签文件的路径
        img_dir_path = os.path.join(img_path_sum, file_name.split('.')[0]) + '/'  # 对应标签文件的图片目录路径
        data = pd.read_csv(label_path)
        valence, arousal = data.iloc[:, 0].values, data.iloc[:, 1].values  # <class 'numpy.ndarray'>
        va_list = np.array([valence.tolist(), arousal.tolist()]).T  # <class 'numpy.ndarray'>
        for _, _, img_files_local in os.walk(img_dir_path):
            img_files = sorted(img_files_local)
            for sub_file in img_files:
                if sub_file == '.DS_Store':
                    continue
                index = convert_num(sub_file)
                if va_list[index - 1][0] != -5 and va_list[index - 1][1] != -5:
                    new_train_img_path = os.path.join(train_img_dir, file_name.split('.')[0] + '_' + sub_file)
                    shutil.copy(os.path.join(img_dir_path, sub_file), new_train_img_path)
                    csv_train_img.append(new_train_img_path)
                    csv_train_v.append(va_list[index - 1][0])
                    csv_train_a.append(va_list[index - 1][1])
                else:
                    pass
annotation_train = pd.DataFrame({'Path': csv_train_img, 'Valence': csv_train_v, 'Arousal': csv_train_a})
annotation_train.to_csv(os.path.join(data_dir, 'train_annotation.csv'), index=False)  # 结果csv存放路径

# 处理验证集
for _, _, files in os.walk(annotation_eval_path):
    sort_files = sorted(files)
    for file_name in sort_files:
        label_path = os.path.join(annotation_eval_path, file_name)  # 标签文件的路径
        img_dir_path = os.path.join(img_path_sum, file_name.split('.')[0]) + '/'  # 对应标签文件的图片目录路径
        data = pd.read_csv(label_path)
        valence, arousal = data.iloc[:, 0].values, data.iloc[:, 1].values  # <class 'numpy.ndarray'>
        va_list = np.array([valence.tolist(), arousal.tolist()]).T  # <class 'numpy.ndarray'>
        for _, _, img_files_local in os.walk(img_dir_path):
            img_files = sorted(img_files_local)
            for sub_file in img_files:
                if sub_file == '.DS_Store':
                    continue
                index = convert_num(sub_file)
                if va_list[index - 1][0] != -5 and va_list[index - 1][1] != -5:
                    new_eval_img_path = os.path.join(eval_img_dir, file_name.split('.')[0] + '_' + sub_file)
                    shutil.copy(os.path.join(img_dir_path, sub_file), new_eval_img_path)
                    csv_eval_img.append(new_eval_img_path)
                    csv_eval_v.append(va_list[index - 1][0])
                    csv_eval_a.append(va_list[index - 1][1])
                else:
                    pass
annotation_eval = pd.DataFrame({'Path': csv_eval_img, 'Valence': csv_eval_v, 'Arousal': csv_eval_a})
annotation_eval.to_csv(os.path.join(data_dir, 'eval_annotation.csv'), index=False)  # 结果csv存放路径
