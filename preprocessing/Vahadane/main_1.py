import numpy as np
import matplotlib.pyplot as plt
import spams
import cv2
import utils
from vahadane import vahadane
from sklearn.manifold import TSNE
from glob import glob
import os
import concurrent.futures
import copy
import time

# 15server_20x
# SOURCE_PATH = '/data15/data15_5/Public/Datasets/stomach/fake_stomach_patch_20x_cancer/*'
# RESULT_PATH = '/data15/data15_5/Public/Datasets/stomach/fake_Vahadane_stomach_patch_20x_cancer'
# TARGET_PATH = './data/target_20x.jpeg'
# 17_1server
# SOURCE_PATH = '/data0/luojing/datasets/stomach/fake_stomach_patch_40x_cancer/*'
# RESULT_PATH = '/data0/luojing/datasets/stomach/fake_Vahadane_stomach_patch_40x_cancer'
# TARGET_PATH = './data/target_40x.jpeg'
# 17_3server
SOURCE_PATH = '/data0/luojing/datasets/stomach/fake_stomach_patch_20x_cancer/*'
RESULT_PATH = '/data0/luojing/datasets/stomach/fake_Vahadane_stomach_patch_20x_cancer'
TARGET_PATH = './data/target_20x.jpeg'
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)

vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=1, getH_mode=0, ITER=50)
vhd.show_config()

vhd.fast_mode = 0
vhd.getH_mode = 0

print(glob(TARGET_PATH))
target_image, aa = utils.read_image(TARGET_PATH)
Wt, Ht = vhd.stain_separate(target_image)
i = 0
slide_num = len(glob(SOURCE_PATH))

def process_image(arg):
    slide_path = arg['slide_path']
    mag = arg['mag']
    patch = arg['patch']
    Wt = arg['Wt']
    Ht = arg['Ht']
    RESULT_PATH = arg['RESULT_PATH']
    slide_name = arg['slide_name']
    path1 = os.path.join(RESULT_PATH, slide_name, mag, patch)
    source_image, p = utils.read_image(os.path.join(slide_path, mag, patch))
    if not os.path.exists(path1):
        print('error file', path1)

        shape = source_image.shape
        if (shape[0] == 512) and (shape[1] == 512) and (p < 210):
            Ws, Hs = vhd.stain_separate(source_image)
            img = vhd.SPCN(source_image, Ws, Hs, Wt, Ht)
            # img = source_image
            cv2.imwrite(path1, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


args = {'slide_path': 0, 'mag': 0, 'patch': 0, 'Wt': Wt, 'Ht': Ht, 'RESULT_PATH': RESULT_PATH, 'slide_name': 0}
for slide_path in glob(SOURCE_PATH):
    start = time.time()
    dicts = []
    i += 1
    print('------speed-------{}/{}'.format(i, slide_num))
    slide_name = os.path.basename(slide_path)
    print(slide_name)
    mag = os.listdir(slide_path)[0]
    print(mag)
    patch_name = os.listdir(os.path.join(slide_path, mag))
    if not os.path.exists(os.path.join(RESULT_PATH, slide_name)):
        os.mkdir(os.path.join(RESULT_PATH, slide_name))
        os.mkdir(os.path.join(RESULT_PATH, slide_name, mag))

    args['slide_path'] = slide_path
    args['mag'] = mag
    args['slide_name'] = slide_name
    for patch in patch_name:
        args['patch'] = patch
        dicts.append(copy.deepcopy(args))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(process_image, dicts)
    print('using time: ', time.time() - start)
