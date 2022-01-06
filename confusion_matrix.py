import os
import torch
import torch.nn as nn
from models.full_model import *
import torchvision.transforms as transforms
from args import args_parser
from val import validation
from datasets import load_datasets
from utils import *
from PIL import Image
import cv2
import numpy as np
from time import time
import pdb
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
os.environ["CUDA_VISIBLE_DEVICES"]="0"

args = args_parser()
best_acc = 0

def cls2label(root, dataset, class_path):
    f = open(os.path.join(root, class_path.replace('dataset', dataset)), 'r')
    line = f.readline()
    cls2label_list ={}
    while line:
        line = line.strip('\n')
        cls, label = line.split(' ')
        cls2label_list[str(cls)] = str(label)
        line = f.readline()
    return cls2label_list

def label2cls(root, dataset, class_path):
    f = open(os.path.join(root, class_path.replace('dataset', dataset)), 'r')
    line = f.readline()
    label2cls_list ={}
    while line:
        line = line.strip('\n')
        cls, label = line.split(' ')
        label2cls_list[str(label)] = str(cls)
        line = f.readline()
    return label2cls_list

def scaling(x):
    max, min = np.max(x), np.min(x)
    x = (x - min) / (max - min)
    return x


def plot_confusion_matrix(dataset, true_list, pred_list, label2cls_list):

    labels = []
    for key, value in label2cls_list.items():
        labels.append(value)
    tick_marks = np.float32(np.array(range(len(labels)))) + 0.5

    cm = confusion_matrix(true_list, pred_list)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()

    fontsize_axis = 4.2
    fontsize_prop = 2.53
    barsize = 5
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_norm[y_val][x_val]
        if c > 0.01:
            color="white" if c > 0.5 else "black"
            plt.text(x_val, y_val, '%0.2f'%(c,), color=color, fontsize=fontsize_prop, va='center', ha='center')

    plt.gca().set_xticks(tick_marks)
    plt.gca().set_yticks(tick_marks)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-', linewidth=0.3)

    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, fontsize=fontsize_axis, rotation=270)
    plt.yticks(xlocations, labels, fontsize=fontsize_axis)

    cb = plt.colorbar(shrink=1.0)
    cb.ax.tick_params(labelsize=barsize)

    plt.tight_layout()

    plt.savefig('./datapath_' + dataset + '.pdf', format='pdf')
    # plt.show()

# attention
if __name__ == '__main__':
    args.dataset = 'dataset' # name
    pred_list = [0,0,0,1,0,1,0,1,1,2,2,1,2,2,2] # the classification result of a batch
    label_list = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2] # corresponding label
    label2cls_list = {'0':'classA','1':'classB','2':'classC'}

    plot_confusion_matrix(args.dataset, label_list, pred_list, label2cls_list)

