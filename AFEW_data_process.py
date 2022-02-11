import json
import os
import shutil

import numpy as np
import pandas as pd


def replace_path(dir_path):
    return dir_path.replace('\\', '/')


def convert_int(file_str):
    return int(file_str.split('.')[0])


currentDir = replace_path(os.getcwd()) + '/'  # 用于返回当前工作目录。  D:\PyCharm\VideoFER
SourceData = os.path.join(currentDir, 'source_data/')  # 源数据文件夹
data_dir = os.path.join(currentDir, 'AFEW_data/')  # 存放处理后的数据文件夹
img_dir = os.path.join(data_dir, 'img/')  # 存放图片文件夹
landmark_dir = os.path.join(data_dir, 'landmark/')  # 存放人脸坐标文件夹
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
if not os.path.exists(landmark_dir):
    os.makedirs(landmark_dir)
print("{}\n{}\n{}\n{}\n{}\n".format(currentDir, SourceData, data_dir, img_dir, landmark_dir))  # 测试文件夹目录路径

valenceList = []  # csv存放Valence的列表
arousalList = []  # csv存放Arousal的列表
frame_pathList = []  # csv存放图片路径的列表
frame_nameList = []  # csv存放图片名的列表
pointList = []  # 存放每张图片的人脸坐标数
pictureSum = 0  # 统计数据集所有图片数目的变量

for root, dirs, files in os.walk(SourceData):
    dirs = sorted(dirs)
    for ID in dirs:
        print(ID + ' directory is being processed……')
        for ID_root, ID_dirs, ID_files in os.walk(os.path.join(SourceData, ID)):  # 进入视频片段文件夹
            ID_files = sorted(ID_files)
            img_len = len(ID_files) - 1  # 一个视频片段下图片总数
            eval_img_len = int(img_len * 0.3)  # 一个视频片段下验证集图片数量
            train_img_len = img_len - eval_img_len  # 一个视频片段下训练集图片数量
            random_idx = np.random.choice(img_len, eval_img_len, replace=False)
            for i in range(len(ID_files)):
                if ID_files[i].split('.')[1] == 'png':
                    pictureSum += 1
                    frame_newName = ID + '_' + ID_files[i]
                    frame_pathList.append(os.path.join(img_dir, frame_newName))
                    frame_nameList.append(frame_newName)
                    shutil.copy(replace_path(os.path.join(SourceData, ID, ID_files[i])), os.path.join(img_dir, frame_newName))
                elif ID_files[i].split('.')[1] == 'json':
                    with open(replace_path(os.path.join(SourceData, ID, ID_files[i]))) as js:
                        json_data = json.load(js)
                        for frameKey in json_data['frames']:  # framekey类型是str, 数值是图片名（00000, 00001, 00002 ……）
                            # get v-a data
                            arousal = json_data['frames'][frameKey]['arousal']
                            valence = json_data['frames'][frameKey]['valence']
                            arousalList.append(float(arousal))
                            valenceList.append(float(valence))

                            # get landmark data
                            pointsPath = os.path.join(landmark_dir, ID + '_' + frameKey + '.txt')
                            if not os.path.exists(os.path.dirname(pointsPath)):
                                os.makedirs(os.path.dirname(pointsPath))
                            with open(pointsPath, "w") as file:
                                pointSize = len(json_data['frames'][frameKey]['landmarks'])
                                pointList.append(pointSize)
                                for x_landmark, y_landmark in json_data['frames'][frameKey]['landmarks']:
                                    file.write(str(x_landmark) + " " + str(y_landmark) + "\n")

annotation = pd.DataFrame({'Path': frame_pathList, 'Name': frame_nameList, 'landmarkNum': pointList, 'Valence': valenceList, 'Arousal': arousalList})
annotation.to_csv(os.path.join(data_dir, 'annotation.csv'), index=False)  # 结果csv存放路径
print(str(pictureSum) + " pictures have been processed, Finish!!!")
