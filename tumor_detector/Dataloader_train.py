import torch.utils.data as data_utils
import torch
from PIL import Image
import numpy as np
import random
import glob
import os
import math
from Utils import utils

Map = {
    'Normal':0,
    'EBV': 1,
    'ELSE': 1
}

utils.seed_torch(seed=0)

class ZSData(data_utils.Dataset):
    def __init__(self, path, path2, transforms=None, padding=0, bi=True):
        
        self.bi = bi

        self.__img_name = []
        self.__label_list = []
        self.__img_list = []

        for tp in ['EBV', 'ELSE']:
            print(tp)
            if not os.path.exists(os.path.join(path, tp)):
                print("No " + os.path.join(path, tp))
            if os.path.exists(os.path.join(path, tp)):
                for sample in os.listdir(os.path.join(path, tp)):
                    
                    if tp == 'EBV':
                        path_arr = glob.glob(os.path.join(path, tp, sample, '*1.jpeg'))
                    elif tp == 'ELSE':
                        path_arr = glob.glob(os.path.join(path, tp, sample, '*1.jpeg'))
                    else:
                        print("Type not found")
                        continue
                        
                    random.shuffle(path_arr)
                    
                    tumor = [Map[tp]] * len(path_arr)

                    self.__label_list.extend(tumor)
                    self.__img_name.extend(path_arr)
                    
        for tp in ['Normal']:
            print(tp)
            if not os.path.exists(os.path.join(path2, tp)):
                print("No " + os.path.join(path2, tp))
            if os.path.exists(os.path.join(path2, tp)):
                for sample in os.listdir(os.path.join(path2, tp)):
                    
                    path_arr = glob.glob(os.path.join(path2, tp, sample, '*0.jpeg'))
                        
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

