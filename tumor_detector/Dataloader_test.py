
import torch.utils.data as data_utils
import torch
from PIL import Image
import numpy as np
import random
import glob
import os
from Utils import utils

Map = {
    'tumor': 1,
    'normal': 0
}
utils.seed_torch(seed=0)

class Dataset_test(data_utils.Dataset):
    def __init__(self, path, transforms=None, padding=0, bi=True):

        self.bi = bi

        self.__img_name = []
        self.__label_list = []
        self.__img_list = []


        path_arr = glob.glob(os.path.join(path, '10.0', '*.jpeg'))
        #tumor = [Map[img_type]] * len(path_arr)
        for patch in path_arr:
            patchname = os.path.basename(patch)
            try:
                if str(patchname.split('.')[0][-1]) == '1':
                    self.__label_list.append(Map['tumor'])
                elif str(patchname.split('.')[0][-1]) == '0':
                    self.__label_list.append(Map['normal'])
            except Exception as e:               
                print('misspatch', path, patchname)            
        #self.__label_list.extend(tumor)
        self.__img_name.extend(path_arr)
        self.__img_name = np.array(self.__img_name)

        if padding != 0 and len(self.__img_name) % padding != 0:
            need = (len(self.__img_name) // padding + 1) * padding - len(self.__img_name)
            indice = np.random.choice(len(self.__img_name), need, replace=True)
            self.__img_name = np.concatenate([self.__img_name, self.__img_name[indice]])

        self.data_transforms = transforms

    def __len__(self):
        return len(self.__img_name)

    def get_label_list(self):
        return self.__label_list

    def __getitem__(self, item):
        img = Image.open(self.__img_name[item])
        img_label = self.__label_list[item]

        if self.data_transforms is not None:
            img = self.data_transforms(img)
        return img, img_label
