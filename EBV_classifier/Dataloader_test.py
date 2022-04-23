import torch.utils.data as data_utils
import torch
from PIL import Image
import jpeg4py as jpeg
import numpy as np
import random
import glob
import os
import math


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


Map = {
    'EBV': 1,
    'ELSE': 0
}
seed_torch(seed=0)


class Dataset_test(data_utils.Dataset):
    def __init__(self, path, img_type, transforms=None, padding=0, bi=True):

        self.bi = bi

        self.__img_name = []
        self.__label_list = []
        self.__img_list = []

        path_arr = glob.glob(os.path.join(path, '*.jpeg'))
        tumor = [Map[img_type]] * len(path_arr)
        self.__label_list.extend(tumor)
        self.__img_name.extend(path_arr)
        self.__img_name = np.array(self.__img_name)

        # if padding != 0 and len(self.__img_name) % padding != 0:
        #     need = (len(self.__img_name) // padding + 1) * padding - len(self.__img_name)
        #     indice = np.random.choice(len(self.__img_name), need, replace=True)
        #     self.__img_name = np.concatenate([self.__img_name, self.__img_name[indice]])

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

