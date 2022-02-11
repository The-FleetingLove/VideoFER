import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class AfewDataset(Dataset):
    def __init__(self, args, mode, val_list, transform=None):
        self.img_path_train = []
        self.img_path_eval = []
        self.valence_train = []
        self.valence_eval = []
        self.arousal_train = []
        self.arousal_eval = []
        self.args = args
        self.mode = mode
        self.list = val_list
        self.transform = transform
        img_num = 30051

        data = pd.read_csv(self.args.annotation_csv)
        img_path_list = data.iloc[:, 0].values  # <class 'numpy.ndarray'>
        valence_list = data.iloc[:, 3].values  # <class 'numpy.ndarray'>
        arousal_list = data.iloc[:, 4].values  # <class 'numpy.ndarray'>

        """划分数据集"""
        for i in range(img_num):
            if i in self.list:
                self.img_path_eval.append(img_path_list[i])
                self.valence_eval.append(valence_list[i])
                self.arousal_eval.append(arousal_list[i])
            else:
                self.img_path_train.append(img_path_list[i])
                self.valence_train.append(valence_list[i])
                self.arousal_train.append(arousal_list[i])

    def __len__(self):
        if self.mode == 'train':
            return len(self.img_path_train)
        else:
            return len(self.img_path_eval)

    def __getitem__(self, item):
        if self.mode == 'train':
            path = self.img_path_train[item]
            valence = self.valence_train[item]  # <class 'numpy.int64'>
            arousal = self.arousal_train[item]
        else:
            path = self.img_path_eval[item]
            valence = self.valence_eval[item]
            arousal = self.arousal_eval[item]
        image = cv2.imread(path)
        # image = image[:, :, ::-1]  # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        return image, np.array([valence, arousal], dtype=np.float32), item
