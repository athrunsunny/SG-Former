from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import json
import random
from glob import glob


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class McDataset(Dataset):
    def __init__(self, data_root, file_list, phase='train', transform=None):
        self.transform = transform
        self.root = os.path.join(data_root, phase)
        class_name = os.listdir(self.root)
        self.labels = {}

        for i in range(len(class_name)):
            self.labels[class_name[i]] = i
        externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
        imgfiles = list()
        for clas_name in class_name:
            for extern in externs:
                imgfiles.extend(glob(self.root + "\\" + clas_name + "\\*." + extern))

        self.A_paths = []
        self.A_labels = []

        for path in imgfiles:
            label = self.labels[path.replace("\\", '/').split('/')[-2]]
            self.A_paths.append(path)
            self.A_labels.append(label)

        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]
        return A, A_label
