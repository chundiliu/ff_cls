import sys
import os
import glob
from PIL import Image
import multiprocessing as mp
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional
from torch.utils.data import DataLoader
from read_data import FFDataset
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import re
from utils import *
import glob

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


DATA_DIR = '/home/chundi/L6_workspace/FutureFertility/data/'
TRAIN_IMAGE_LIST = 'train_ff_labels.txt'
VALID_IMAGE_LIST = 'valid_ff_labels.txt'
TEST_IMAGE_LIST = 'test_ff_labels.txt'
BATCH_SIZE = 16
N_CLASSES = 1
MAX_ITER = 200

def generate_label_file():
    splits = ['train', 'valid', 'test']

    for split in splits:
        file_list = glob.glob(DATA_DIR + split + '/*/*')
        with open(split + '_ff_labels.txt', 'w') as f:
            for line in file_list:
                eles = line.split('/')
                valid_eles = eles[-3:]
                if valid_eles[1] == 'pos':
                    label = 1
                elif valid_eles[1] == 'neg':
                    label = 0
                else:
                    print('something goes wrong!')

                s = '/'.join(valid_eles) + ' ' + str(label) + '\n'
                f.write(s)



def train():

    #cudnn.benchmark = True
    model = DenseNet121(N_CLASSES).cuda()
    model = torch.nn.DataParallel(model).cuda()
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = FFDataset(data_dir=DATA_DIR,
                                     image_list_file=TRAIN_IMAGE_LIST,
                                     transform=transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.RandomAffine(degrees=15, translate=None, scale=(0.8, 1.2), shear=5),
                                         transforms.RandomCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomRotation(15),
                                         transforms.ToTensor(),
                                         normalize
                                     ]))

    valid_dataset = FFDataset(data_dir=DATA_DIR,
                                     image_list_file=VALID_IMAGE_LIST,
                                     transform=transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.TenCrop(224),
                                         transforms.Lambda
                                         (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                         transforms.Lambda
                                         (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                     ]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=8, pin_memory=True)

    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=8, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, mode='min')

    loss = torch.nn.BCELoss()

    lossMIN = 9999

    launchTimestamp = time.strftime("%d%m%Y") + '-' + time.strftime("%H%M%S")

    for epochID in range(0, MAX_ITER):
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        timestampSTART = timestampDate + '-' + timestampTime
        trainLoss = epochTrain(model, train_loader, optimizer, scheduler, MAX_ITER, N_CLASSES, loss)
        lossVal, losstensor, accuracy, auc, best_proba, best_mcc = epochVal(model, valid_loader, optimizer,
                                                                                   scheduler, MAX_ITER,
                                                                                   N_CLASSES, loss, False)

        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        timestampEND = timestampDate + '-' + timestampTime

        scheduler.step(losstensor.item())

        if lossVal < lossMIN:
            lossMIN = lossVal
            torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
                        'optimizer': optimizer.state_dict()}, '.dense121_ff_256_avgpool-' + launchTimestamp + '.pth.tar')
            print('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(
                lossVal) + ' trainLoss: {} acc: {}, auc: {}, best_mcc: {}'.format(trainLoss, accuracy, auc, best_proba, best_mcc))
        else:
            print('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal) +
                  ' trainLoss: {} acc: {}, auc: {}, p_500: {}, best_mcc: {}'.format(trainLoss, accuracy, auc, best_proba, best_mcc))

def epochTrain(model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):

    model.train()

    lossVal = 0
    lossValNorm = 0

    for batchID, (input, target) in enumerate(dataLoader):
        target = target.cuda(async=True)

        varInput = torch.autograd.Variable(input)
        varTarget = torch.autograd.Variable(target)
        varOutput = model(varInput)

        lossvalue = loss(varOutput, varTarget)

        lossVal += lossvalue.item()
        lossValNorm += 1

        optimizer.zero_grad()
        lossvalue.backward()

        optimizer.step()

    outLoss = lossVal / lossValNorm

    return outLoss

def epochVal(model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, save_pred = False):

    model.eval()

    lossVal = 0
    lossValNorm = 0

    losstensorMean = 0

    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()


    for i, (input, target) in enumerate(dataLoader):
        target = target.cuda(async=True)

        gt = torch.cat((gt, target), 0)

        bs, n_crops, c, h, w = input.size()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input.view(-1, c, h, w).cuda())

            varTarget = torch.autograd.Variable(target)
            varOutput = model(input_var)
            varOutput = varOutput.view(bs, n_crops, -1).mean(1)

            pred = torch.cat((pred, varOutput.data), 0)

            losstensor = loss(varOutput, varTarget)
            losstensorMean += losstensor

            lossVal += losstensor.item()
            lossValNorm += 1

    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()

    if save_pred:
        np.save('valid_pred', pred)

    accuracy = accuracy_score(gt, pred > 0.50)
    auc = roc_auc_score(gt, pred)

    best_proba, best_mcc, _ = eval_mcc(gt.squeeze(), pred.squeeze(), True)

    outLoss = lossVal / lossValNorm
    losstensorMean = losstensorMean / lossValNorm

    return outLoss, losstensorMean, accuracy, auc, best_proba, best_mcc


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True, drop_rate=0.0)
        num_ftrs = self.densenet121.classifier.in_features
        print(num_ftrs)
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.densenet121(x)
        return x


if __name__ == '__main__':
    generate_label_file()
    train()