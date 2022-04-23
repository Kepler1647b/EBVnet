import torch.utils.data as data_utils
from PIL import Image
import numpy as np
import random
import glob
import os
from Utils import utils

utils.seed_torch(seed=0)
Map = {
    'EBV': 1,
    'ELSE': 0
}


class ZSData(data_utils.Dataset):
    def __init__(self, path, transforms=None, padding=0, bi=True):
        
        self.bi = bi
        self.__img_name = []
        self.__label_list = []
        self.__img_list = []
        
        EBV_num = 0
        ELSE_num = 0
        limit_num = 1000   # 无限制
        
        for tp in ['EBV', 'ELSE']:
            print(tp)
            if not os.path.exists(os.path.join(path, tp)):
                print("No " + os.path.join(path, tp))
            if os.path.exists(os.path.join(path, tp)):
                for sample in os.listdir(os.path.join(path, tp)):

                    if tp == 'EBV':
                        if EBV_num < limit_num:
                            EBV_num += 1
                            path_arr = glob.glob(os.path.join(path, tp, sample, '*1.jpeg'))
                        else:
                            continue
                    elif tp == 'ELSE':
                        if ELSE_num < limit_num:
                            ELSE_num += 1
                            path_arr = glob.glob(os.path.join(path, tp, sample, '*1.jpeg'))
                        else:
                            continue
                    else:
                        print("Type not found")
                        continue
                        
                    random.shuffle(path_arr)
                    tumor = [Map[tp]] * len(path_arr)

                    self.__label_list.extend(tumor)
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

