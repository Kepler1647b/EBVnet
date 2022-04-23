# -*- coding: utf-8 -*
import os
import re
from argparse import ArgumentParser
import sys
import gc
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tumor_detector.Dataloader_test import Dataset_test
from pandas import DataFrame
from sklearn.metrics import auc, confusion_matrix, roc_curve
import Utils.config_stomach as CONFIG
from Utils import utils
sys.path.append('/data0/zihan/Datasets')
from Model.create_model import create_model
plt.switch_backend('agg')


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_index(path):
    path = os.path.basename(path)
    print(path)
    y, x = list(map(int, re.findall(r'\d+', path)[-2: ]))
    return x, y


Map = {
    'normal': 0,
    'tumor': 1,
    'EBV': -1,
    'ELSE': -2
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--ModelPath", dest='model_path', type=str)
    parser.add_argument("--DataPath", dest='data_path', type=str)
    parser.add_argument("--DataPath2", dest='data_path2', default='', type=str)
    parser.add_argument("--DeviceId", dest='device_id', default='6', type=str)
    parser.add_argument("--Model", dest='model', default='resnet50', type=str)
    parser.add_argument("--Foldn", dest='foldn', default='1', type=str)
    parser.add_argument("--ResultPath", dest='resultpath', default='02_28', type=str)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

    if args.device_id is not None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    if torch.cuda.device_count() == 2:
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        model = torch.nn.DataParallel(model, device_ids=[0,1])
    
    elif torch.cuda.device_count() == 4:
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])

    model.eval()

    test_path = args.data_path
    path2 = args.data_path2
    
    test_cases = set(map(lambda x: x, os.listdir(os.path.join(test_path))))

    slides, ground_truth, aver, counts = [], [], [], []
    all_labels, all_predicts, all_values = [], [], []
    cases, cases_labels, cases_predicts, cases_values, cases_aver = [], [], [], [], []
    empty_slide = []
    empty_case = []

    all_labeldics = np.array([])
    all_valuedics = np.array([])
    all_preddics = np.array([])

    slide_names1 = []
    auc1 = []
    acc1 = []

    cnt = 0

    for slidename in test_cases:
        gc.collect()
        labeldic = []
        cnt += 1
        print('-------------speed: {}/{}--------------'.format(cnt, len(test_cases)))
        groundtruth = 'tumor'
        patchs_aver = np.zeros([2, 1])
        patchs_count = np.zeros([2, 1])
        ground_truth.append(groundtruth)
        print('work')
        # patchs = glob(os.path.join(test_path, slidename, '10.0', '*.jpeg'))
        slidepath = os.path.join(test_path, slidename)

        test_dataset = Dataset_test(path=slidepath, transforms=utils.transform_test, padding=0, bi=True)
        sample_num = test_dataset.__len__()
        print('sample_num', sample_num)

        if sample_num == 0:
            aver.append([-1, -1])
            counts.append([-1, -1])
            continue

        rlt_aver = np.zeros([2, sample_num])
        rlt_count = np.zeros([2, sample_num])
        bs = 32
        test_loader = DataLoaderX(test_dataset, batch_size=bs, num_workers=4, pin_memory=True, persistent_workers=True)
        cur_patch = 0
        bs = 8
        current_patch = 0

        for (inputs, labels) in test_loader:
            tmp = len(labels)
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            value, preds = torch.max(outputs, 1)
            outputs, v, p = outputs.detach().cpu().numpy(), value.detach().cpu().numpy(), preds.detach().cpu().numpy()

            for j in range(tmp):
                for index in range(2):
                    rlt_aver[index][current_patch + j] = outputs[j][index]
                    rlt_count[index][current_patch + j] = (preds[j] == index)
                
            current_patch += tmp
        average = rlt_aver.mean(axis = 1)  # 预测值的平均
        count = rlt_count.mean(axis = 1)  # 阴性阳性预测数量的平均
        valuedic = rlt_aver[1][:]
        preddic = rlt_count[1][:]
        labeldic = test_dataset.get_label_list()
        labeldic = np.array(labeldic)
        print(labeldic)
        all_labeldics = np.concatenate((all_labeldics, labeldic))
        print(all_labeldics)
        all_valuedics = np.concatenate((all_valuedics, valuedic))
        all_preddics = np.concatenate((all_preddics, preddic))
        print("Acc for %s: " % slidename, (labeldic == preddic).sum() / len(labeldic))
        fpr, tpr, theshold = roc_curve(labeldic, valuedic, pos_label=1)
        print('AUC for %s: ' % slidename, auc(fpr, tpr))
        slide_names1.append(slidename)
        acc1.append((labeldic == preddic).sum() / len(labeldic))
        auc1.append(auc(fpr, tpr))
        patchs_aver = np.concatenate((patchs_aver, rlt_aver), axis = 1)
        print(patchs_aver.shape)
        patchs_count = np.concatenate((patchs_count, rlt_count), axis = 1)
        print(average, count)
        aver.append(average)
        counts.append(count)
        all_labels.append(Map[groundtruth])
        all_predicts.append(np.argmax(average))
        all_values.append(average[1])

        # print("label: ",Map[groundtruth])
        # print("predict: ", np.argmax(average))

        if patchs_aver.shape[1] == 1:
            empty_case.append(slidename)
            continue
        else:
            cases.append(slidename)
            patchs_aver = np.delete(patchs_aver, 0, axis = 1)
            patchs_count = np.delete(patchs_count, 0, axis = 1)
            case_average = patchs_aver.mean(axis = 1)
            case_count = patchs_count.mean(axis = 1)
            cases_aver.append(case_average)
            cases_labels.append(Map['tumor'])
            cases_predicts.append(np.argmax(case_average))
            cases_values.append(case_average[1])
            print('case %s finish' % slidename)


        # print(patchs)
    all_labels = np.array(all_labels)
    all_predicts = np.array(all_predicts)
    all_values = np.array(all_values)
    eerthreshold = utils.eer_threshold(all_labeldics, all_valuedics)
    # Loss = running_loss / len(all_labeldics)
    Acc1 = (all_preddics == all_labeldics).sum()
    Acc = Acc1 / len(all_labeldics)
    TP = ((all_preddics == 1) & (all_labeldics == 1)).sum()
    TN = ((all_preddics == 0) & (all_labeldics == 0)).sum()
    FN = ((all_preddics == 0) & (all_labeldics == 1)).sum()
    FP = ((all_preddics == 1) & (all_labeldics == 0)).sum()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    noatresia_p = TN / (TN + FN)
    noatresia_r = TN / (TN + FP)
    conf_matrix = confusion_matrix(all_labeldics, all_preddics, labels=list(range(CONFIG.NUM_CLASSES)))#混淆矩阵
    avg_acc = utils.print_conf_matrix(conf_matrix, CONFIG.CLASSES1)
    # print('%s loss:' %  Loss)
    print('Acc: %s ' %  Acc)
    
    sensitivity = TP / (TP + FN)
    specitivity = TN / (FP + TN)
    precision = (TP) / (TP + FP)
    print('precision: ', precision)
    print('sensitivity : %s' % sensitivity)
    print('specificity : %s' % specitivity)
    print('PPV: %s' % p)
    print('NPV: %s' % noatresia_p)


    fpr, tpr, theshold = roc_curve(all_labeldics, all_valuedics, pos_label=1)
    print('AUC: ', auc(fpr, tpr))
    AUC = auc(fpr, tpr)

    auc_95_ci, sensi_95, speci_95 = utils.print_result(all_labeldics, all_valuedics, all_preddics)

    df = DataFrame(
        {
        'TP': TP,
        'TN': TN,
        'FN': FN,
        'FP': FP,
        'AUC': AUC,
        'auc_95_ci': auc_95_ci,
        'sensitivity': sensitivity,
        'sensi_95': sensi_95,
        'specitivity': specitivity,
        'speci_95': speci_95,
        'precision': precision}
    )
    save_path = os.path.join(args.resultpath, args.foldn)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    df.to_csv(os.path.join(save_path, 'slidelevel_tumordetector_result_fold' + args.foldn + '.csv'))


