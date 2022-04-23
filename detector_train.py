import os
import time
import numpy as np
import Utils.config_stomach as CONFIG
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, f1_score
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from tumor_detector.Dataloader_train import ZSData
from Model.create_model import create_model
from Utils import utils
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def run(model, dataloaders, phase, device, criterion, optimizer):
    
    if phase == 'train':
        model.train()
    elif phase == 'valid':
        model.eval()
    else:
        raise Exception("Error phase")
    running_loss = 0.0
    running_corrects = 0
    all_labels = np.array([])
    all_predicts = np.array([])
    all_values = np.array([])
    
    for i, (inputs, labels) in enumerate(tqdm(dataloaders)):

        if (i != 0) and (i % 200) == 0:
            fpr, tpr, theshold = roc_curve(all_labels, all_values, pos_label=1)
            print('Process<{}> [{}/{}] Current acc: {:.5f}  AUC : {:.5f}'.format(
                phase, i, len(dataloaders), (all_labels == all_predicts).sum() / len(all_predicts), auc(fpr, tpr)
            ))
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer.zero_grad()

        with torch.torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            value = outputs[:,1]
            _loss = criterion(outputs, labels)
            
            #update model and optimizer
            if phase == 'train':
                _loss.backward()
                optimizer.step()

        # update train/valid diagnostics
        running_loss += _loss.item() * inputs.size(0)

        all_labels = np.concatenate((all_labels, labels.cpu().numpy()))
        all_predicts = np.concatenate((all_predicts, preds.cpu().numpy()))
        all_values = np.concatenate((all_values, value.detach().cpu().numpy()))

    # print train/valid diagnostics
    Loss = running_loss / (len(all_labels))
    Acc1 = (all_labels == all_predicts).sum()
    Acc2 = len(all_predicts)
    Acc = Acc1 / Acc2
    TP = ((all_predicts == 1) & (all_labels == 1)).sum()
    TN = ((all_predicts == 0) & (all_labels == 0)).sum()
    FN = ((all_predicts == 0) & (all_labels == 1)).sum()
    FP = ((all_predicts == 1) & (all_labels == 0)).sum()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    conf_matrix = confusion_matrix(all_labels, all_predicts, labels=list(range(CONFIG.NUM_CLASSES)))
    
    print("%s confusion matrix:" % phase.capitalize())
    avg_acc = utils.print_conf_matrix(conf_matrix, CONFIG.CLASSES)
    
    Writer.add_scalar('%s/Loss' % phase.capitalize(), Loss, global_epoch) ##
    Writer.add_scalar('%s/Acc'  % phase.capitalize(), Acc, global_epoch) ##
    

    fpr, tpr, theshold = roc_curve(all_labels, all_values, pos_label=1)
    print('AUC', auc(fpr, tpr))
    
    Writer.add_scalar('{}/AUC'.format(phase.capitalize()), auc(fpr, tpr), global_epoch)
    Writer.add_scalar('%s/EBV_Precision' % phase.capitalize(), p, global_epoch)
    Writer.add_scalar('%s/EBV_Recall' % phase.capitalize(), r, global_epoch)
    Writer.add_scalar('%s/EBV_F1' % phase.capitalize(), F1, global_epoch)
    Writer.add_figure('{}[{}]'.format(phase.capitalize(), global_epoch),figure=utils.plot_confusion_matrix(
        conf_matrix, classes=CONFIG.CLASSES, title='Confusion Matrix'), global_step=global_epoch)
    
    return Loss, p, r, F1, Acc, auc(fpr, tpr), conf_matrix


def start_train(train_loader, valid_loader, model, device, criterion, optimizer, scheduler, num_epochs):
    
    best_F1 = .0
    best_auc = .0
    best_loss = 1000000 
    
    for epoch in range(1, num_epochs + 1):
        global global_epoch
        global_epoch = epoch
        
        print('\n##### Epoch [{}/{}]'.format(epoch, num_epochs))
        
        print('\n####### Train #######')
        train_loss, t_p, t_r, t_F1, train_acc, train_auc, train_cm = run(model, train_loader, 'train', device, criterion, optimizer)
        
        print('\n####### Valid #######')
        val_loss, v_p, v_r, v_F1, val_acc, val_auc, val_cm = run(model, valid_loader, 'valid', device, criterion, optimizer)
        
        ##
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        Writer.add_scalar('Learning Rate', current_lr, epoch) ##

        print('Epoch {} with lr {:.15f}: t_loss: {:.4f} t_acc: {:.4f} v_loss:{:.4f} v_acc: {:.4f}\n'.format(epoch, current_lr, train_loss, train_acc, val_loss, val_acc))
        print('EBV: t_precision: {:.4f}, t_recall: {:.4f}, t_F1: {:.4f}'.format(t_p, t_r, t_F1))
        print('EBV: v_precision: {:.4f}, v_recall: {:.4f}, v_F1: {:.4f}'.format(v_p, v_r, v_F1))
        if v_F1 > best_F1:
            best_F1 = v_F1
            torch.save(model.state_dict(), os.path.join(
                checkpoints, 'bestF1.pt')
            )
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(
                checkpoints, 'bestloss.pt')
            )
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(
                checkpoints, 'bestauc.pt')
            )
            
        
    torch.save(model.state_dict(), os.path.join(
        checkpoints, 'FINAL.pt')
    )


def prepare_data(dataset_path, dataset_path2, batch_size):
    
    train_dataset = ZSData(os.path.join(dataset_path, 'train'), os.path.join(dataset_path2, 'train'), transforms=utils.transform_train, bi=True, padding=0)
    valid_dataset = ZSData(os.path.join(dataset_path, 'valid'), os.path.join(dataset_path2, 'valid'), transforms=utils.transform_valid, bi=True, padding=0)
    
    nclasses = len(CONFIG.CLASSES)
    weights, weight_per_class = utils.make_weights_for_balanced_classes(train_dataset.get_label_list(), nclasses)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) 

    print("batch_size", batch_size)
    train_loader = DataLoaderX(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    valid_loader = DataLoaderX(valid_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
        
    return train_loader, valid_loader, weight_per_class


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--TrainFolder", dest='train_folder', default='./data/stomach/Vahadane_stomach_split_10x_cancer', type=str)
    parser.add_argument("--TrainFolder2", dest='train_folder2', default='/data0/luojing/datasets/stomach/Vahadane_stomach_split_10x_normal', type=str)
    parser.add_argument("--FoldN", dest='foldn', default=1, type=int)
    parser.add_argument("--Loss", dest='loss_func', default='cross', type=str, help='the type of loss function')
    parser.add_argument("--NumEpoch", dest='num_epochs', default=50, type=int, help='training epoch')
    parser.add_argument("--Seed", dest='seed', default=0, type=int, help='setting the random seed of this file')
    parser.add_argument("--Model", dest='model', default='resnet50', type=str, help='the name of the used model')
    parser.add_argument("--LearningRate", dest='learning_rate', default=0.001, type=float, help='the learning rate of training')
    parser.add_argument("--BatchSize", dest='batch_size', default=64, type=int, help='the batch size of training')
    parser.add_argument("--WeightDecay", dest='weight_decay', default=0.0005, type=float, help='learning rate decay')     # learning rate decay
    parser.add_argument("--DeviceId", dest='device_id', default='0', type=str, help='choose the GPU id to use')
    parser.add_argument("--Comment", dest='comment', type=str, help='the pa')
    parser.add_argument("--Pretrain", dest='pretrain', action="store_true")
    
    args = parser.parse_args()
    
    utils.seed_torch(args.seed)
    
    loss_func = args.loss_func
   
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
 
    now = time.strftime("%Y_%m_%d_", time.localtime())
    
    global checkpoints
    checkpoints = os.path.join(CONFIG.CHECK_POINT, 
        now+"BS"+str(args.batch_size)+","+args.comment+",seed"+str(args.seed)+",fold"+str(args.foldn))
    
    print('Summary write in %s' % checkpoints)
    
    Writer = SummaryWriter(log_dir=checkpoints)
    
    train_loader, valid_loader, weight_per_class = prepare_data(os.path.join(args.train_folder, str(args.foldn)), os.path.join(args.train_folder2, str(args.foldn)), args.batch_size)

    print(len(CONFIG.CLASSES), 'classes:', CONFIG.CLASSES)
    
    print('num train images %d x %d' % (len(train_loader), args.batch_size))
    
    print('num val images %d x %d' % (len(valid_loader), args.batch_size))
    
    print("CUDA is_available:", torch.cuda.is_available())
    
    if args.device_id is not None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    print('\n### Build model %s' % args.model)
    
    model = create_model(args.model, args.pretrain)
    model = model.to(device)
    if torch.cuda.device_count() == 2:
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        model = torch.nn.DataParallel(model, device_ids=[0,1])
    elif torch.cuda.device_count() == 4:
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])


    # cross entropy loss
    if loss_func == 'cross':
        weight_per_class = torch.Tensor(weight_per_class).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_per_class)
        
    optimizer = optim.SGD(model.parameters(),
                            momentum = 0.9,
                            lr=args.learning_rate,
                            weight_decay=args.weight_decay,
                            nesterov=True)
                            
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.)

    start_train(train_loader, valid_loader, model, device, criterion,
                optimizer, scheduler, args.num_epochs)

