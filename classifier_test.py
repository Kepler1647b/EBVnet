import os
import re
import warnings
import numpy as np
from pandas import DataFrame
import Utils.config_stomach as CONFIG
from argparse import ArgumentParser
import torch.nn.functional as F
import gc
from glob import glob
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from EBV_classifier.Dataloader_test import Dataset_test
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from Model.create_model import create_model
from Utils import utils
from math import ceil

plt.switch_backend('agg')
warnings.filterwarnings('ignore')

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
# pass

def get_index(path):
    path = os.path.basename(path)
    y, x = list(map(int, re.findall(r'\d+', path)[-3:-1]))
    return x, y


Map = {
    'EBV': 1,
    'ELSE': 0,
}

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--ModelPath", dest='model_path', type=str, help='the trained model path')
    parser.add_argument("--DataPath", dest='data_path', default='/data15/data15_5/Public/Datasets/stomach/Vahadane_stomach_split_10x_cancer',
                        type=str, help='test data path')
    parser.add_argument("--FoldN", dest='foldn', type=int, help='which fold in the test data')
    parser.add_argument("--DeviceId", dest='device_id', type=str, help='choose the GPU id')
    parser.add_argument("--Model", dest='model', default='resnet50', type=str, help='the training model')
    parser.add_argument("--Seed", dest='seed', default=0, type=int, help='the random seed in used in this file')

    parser.add_argument("--SaveFile", dest='saveFile', default='seed_result', type=str, help='the path to save the test result')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    if args.device_id is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')

    model = create_model(args.model, False)
    state_dict = torch.load(args.model_path)
    for key in state_dict.keys():
        if 'module.' in key:
            print('parallel')
            model = torch.nn.DataParallel(model)
            break
    model.load_state_dict(state_dict, True)
    model = model.to(device)
    model.eval()

    path = args.data_path
    test_path = os.path.join(path, str(args.foldn), 'test')
    test_cases = {t: set(map(lambda x: x.split('_')[0], os.listdir(os.path.join(test_path, t))))
                  for t in os.listdir(test_path)}

    # test
    slides, ground_truth, aver, count = [], [], [], []
    all_labels, all_predicts, all_values = [], [], []

    for t in ['ELSE', 'EBV']:
        test_cases[t] = sorted(test_cases[t])
        cnt = 0
        for case_name in test_cases[t]:
            print('--------speed-----{}/{}'.format(cnt, len(test_cases[t])))
            gc.collect()
            cnt += 1
            print("cnt:{}  type_num:{}".format(cnt, len(test_cases[t])))
            print(case_name)
            slides.append(case_name)

            groundtruth = t
            print(groundtruth)
            ground_truth.append(groundtruth)
            sample_path = os.path.join(test_path, t, case_name)
            patches = glob(os.path.join(test_path, t, case_name, '*1.jpeg'))
            patches = patches[:ceil(len(patches) / 1)]
            print('sample_path:', sample_path)
            test_dataset = Dataset_test(path=sample_path, img_type=t, transforms=utils.transform_test, padding=0, bi=True)
            sample_num = test_dataset.__len__()
            print('sample_num', sample_num)

            if sample_num == 0:
                aver.append([-1, -1])
                count.append([-1, -1])
                continue

            rlt_aver = np.zeros([2, sample_num])
            rlt_count = np.zeros([2, sample_num])
            bs = 32
            test_loader = DataLoaderX(test_dataset, batch_size=bs, num_workers=4, pin_memory=True, persistent_workers=True)
            cur_patch = 0
            for (inputs, labels) in test_loader:
                tmp = len(labels)
                inputs = inputs.to(device)
                outputs = model(inputs)
                outputs = F.softmax(outputs, dim=1)
                value, preds = torch.max(outputs, 1)
                outputs, v, p = outputs.detach().cpu().numpy(), value.detach().cpu().numpy(), preds.detach().cpu().numpy()
                for j in range(tmp):
                    for index in range(2):
                        rlt_aver[index][cur_patch + j] = outputs[j][index]
                        rlt_count[index][cur_patch + j] = p[j] == index
                cur_patch += tmp
            average = rlt_aver.mean(axis=1)
            counts = rlt_count.mean(axis=1)
            print(average, counts)

            # heatmap
            # position = [get_index(x) for x in patches]
            #
            # width  = max([x for x,y in position]) + 1
            # height = max([y for x,y in position]) + 1
            #
            # fig = plt.figure(figsize=(height, width))
            #
            # result = np.ones([width, height]) * -1
            # for i, (x, y) in enumerate(position):
            #     result[x][y] = rlt_aver[0][i]
            #
            # if groundtruth == 'EBV':
            #     s = sns.heatmap(result, cmap="RdYlBu", vmax=1.001, vmin=-0.001, mask=(result==-1), xticklabels=False, yticklabels=False, cbar=False)
            # else:
            #     s = sns.heatmap(result, cmap="RdYlBu", vmax=1.001, vmin=-0.001, mask=(result==-1), xticklabels=False, yticklabels=False, cbar=False)
            #
            # plt.savefig(os.path.join(os.path.dirname(args.model_path),
            #      '{}_{}_{}.jpeg'.format(case_name, groundtruth, 'EBV' if average[1] > 0.5 else 'ELSE')))
            # plt.close()

            aver.append(average)
            count.append(counts)

            all_labels.append(Map[groundtruth])
            print("label: ", Map[groundtruth])
            all_predicts.append(np.argmax(average))
            print("predict: ", np.argmax(average))
            all_values.append(average[1])

    print("Threshold 0.5:")
    print("Confusion matrix:")
    conf_matrix = confusion_matrix(all_labels, all_predicts, labels=list(range(CONFIG.NUM_CLASSES)))
    utils.print_conf_matrix(conf_matrix, CONFIG.CLASSES)

    all_labels = np.array(all_labels)
    all_predicts = np.array(all_predicts)
    all_values = np.array(all_values)
    TP = ((all_predicts == 1) & (all_labels == 1)).sum()
    TN = ((all_predicts == 0) & (all_labels == 0)).sum()
    FN = ((all_predicts == 0) & (all_labels == 1)).sum()
    FP = ((all_predicts == 1) & (all_labels == 0)).sum()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    print('EBV: precision: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(p, r, F1))

    print("Acc: ", (all_labels == all_predicts).sum() / len(all_predicts))
    fpr, tpr, theshold = roc_curve(all_labels, all_values, pos_label=1)
    print('AUC: ', auc(fpr, tpr))

    df = DataFrame({
        'Slide': slides,
        'Type': ground_truth,
        'Averaging': aver,
        'Counting': count
    })
    save_path = os.path.join(args.saveFile, 'inner_seed' + str(args.seed) + '_fold' + str(args.foldn) + '.csv')
    # df.to_csv(os.path.join(os.path.dirname(args.model_path), os.path.dirname(args.model_path)[-10:] + 'stomach_result_' + os.path.basename(args.model_path) +'.csv'))
    df.to_csv(save_path)