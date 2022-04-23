import numpy as np
import matplotlib.pyplot as plt
import spams
import cv2
import utils
from vahadane import vahadane
from sklearn.manifold import TSNE
from glob import glob
import os

SOURCE_PATH = '/data15/data15_5/Public/Datasets/stomach/fake_stomach_patch_20x_cancer/*'
TARGET_PATH = './data/target_20x.jpeg'
RESULT_PATH = '/data15/data15_5/Public/Datasets/stomach/fake_Vahadane_stomach_patch_20x_cancer'
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)


vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=1, getH_mode=0, ITER=50)
vhd.show_config()

vhd.fast_mode=0;vhd.getH_mode=0;

print(glob(TARGET_PATH))
target_image = utils.read_image(TARGET_PATH)
Wt, Ht = vhd.stain_separate(target_image)
i = 0
slide_num = len(glob(SOURCE_PATH))
for slide_path in glob(SOURCE_PATH):
    
    i += 1
    print('------speed-------{}/{}'.format(i,slide_num))
    slide_name = os.path.basename(slide_path)
    print(slide_name)
    mag = os.listdir(slide_path)[0]
    print(mag)
    patch_name = os.listdir(os.path.join(slide_path, mag))
    if os.path.exists(os.path.join(RESULT_PATH, slide_name)):
        continue
    os.mkdir(os.path.join(RESULT_PATH, slide_name))
    os.mkdir(os.path.join(RESULT_PATH, slide_name, mag))
    for patch in patch_name:
        source_image = utils.read_image(os.path.join(slide_path, mag, patch))
        Ws, Hs = vhd.stain_separate(source_image)
        img = vhd.SPCN(source_image, Ws, Hs, Wt, Ht)
        cv2.imwrite(os.path.join(RESULT_PATH, slide_name, mag, patch), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
