from io import BytesIO
# import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import data.util as Util
import scipy
import scipy.io
import os
from torch.utils.data import DataLoader

import numpy as np

def make_dataset2(image_root):
    imgs = []
    list = os.listdir(image_root)
    for a in list:
        if a == 'normal':
            path = os.listdir(image_root + '/' + a)
            for b in path:
                path_true = image_root + '/' + a + '/' + b
                imgs.append((path_true, 0))
        if a == 'sicken':
            path = os.listdir(image_root + '/' + a)
            for b in path:
                path_true = image_root + '/' + a + '/' + b
                imgs.append((path_true, 1))
    random.shuffle(imgs)
    return imgs

def make_dataset(image_root):

    imgs = []
    list = os.listdir(image_root)
    for a in list:
        if a == '0':
            path = os.listdir(image_root + '/' + a)
            for b in path:
                path_true = image_root + '/' + a + '/' + b
                imgs.append((path_true, 0))
        if a == '1':
            path = os.listdir(image_root + '/' + a)
            for b in path:
                path_true = image_root + '/' + a + '/' + b
                imgs.append((path_true, 1))
        if a == '2':
            path = os.listdir(image_root + '/' + a)
            for b in path:
                path_true = image_root + '/' + a + '/' + b
                imgs.append((path_true, 2))
        if a == '3':
            path = os.listdir(image_root + '/' + a)
            for b in path:
                path_true = image_root + '/' + a + '/' + b
                imgs.append((path_true, 3))
        if a == '4':
            path = os.listdir(image_root + '/' + a)
            for b in path:
                path_true = image_root + '/' + a + '/' + b
                imgs.append((path_true, 4))
        if a == '5':
            path = os.listdir(image_root + '/' + a)
            for b in path:
                path_true = image_root + '/' + a + '/' + b
                imgs.append((path_true, 5))
        if a == '6':
            path = os.listdir(image_root + '/' + a)
            for b in path:
                path_true = image_root + '/' + a + '/' + b
                imgs.append((path_true, 6))
        if a == '7':
            path = os.listdir(image_root + '/' + a)
            for b in path:
                path_true = image_root + '/' + a + '/' + b
                imgs.append((path_true, 7))
        if a == '8':
            path = os.listdir(image_root + '/' + a)
            for b in path:
                path_true = image_root + '/' + a + '/' + b
                imgs.append((path_true, 8))
        if a == '9':
            path = os.listdir(image_root + '/' + a)
            for b in path:
                path_true = image_root + '/' + a + '/' + b
                imgs.append((path_true, 9))
    random.shuffle(imgs)
    return imgs

class ImageDataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        self.res = resolution
        self.data_len = data_len
        self.split = split

        # self.path = Util.get_paths_from_images(dataroot)
        #
        self.path = make_dataset2(dataroot)
        self.dataset_len = len(self.path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img = Image.open(self.path[index]).convert("RGB")
        img = img.resize((256, 256))
        img = Util.transform_augment(img, split=self.split, min_max=(-1, 1), res=self.res)
            
        return {'img': img,  'Index': index}


class ImageDataset2(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        self.res = resolution
        self.data_len = data_len
        self.split = split
        self.path = make_dataset2(dataroot)
        self.dataset_len = len(self.path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img = Image.open(self.path[index][0]).convert("RGB")
        img = img.resize((256, 256))
        img = Util.transform_augment(img, split=self.split, min_max=(-1, 1), res=self.res)
        label = torch.tensor(self.path[index][1], dtype=torch.long)
        return {'A': img, 'B': img, 'L': label, 'Index': index}

if __name__ == '__main__':
    liver_dataset = ImageDataset2('D:\扩散模型代码\ddpm-cd-master 2\ddpm-cd-master\datasets\mnist_data_test')
    # print(liver_dataset.imgs)
    dataloaders = DataLoader(liver_dataset, batch_size=2, shuffle=True, num_workers=0)
    for custep, train_1 in enumerate(dataloaders):
        print(custep, train_1)
        print("label:", train_1['L'], train_1['L'].dtype)
        break
