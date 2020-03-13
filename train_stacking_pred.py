import os, time, sys
import numpy as np
# np.random.bit_generator = np.random._bit_generator
import torch
from torchvision import transforms, datasets
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import sklearn.metrics
import random
from datetime import datetime
from tqdm import tqdm

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

from torch_lr_finder import LRFinder
from One_Cycle_Policy import OneCycle
# from class_balanced_loss import get_cb_loss
from apex import amp
import cv2

device = "cuda"

# train_images = np.load("./train_images_invert_0204.npy")
# train_labels = np.load("./train_labels_shuffle_0204.npy")

train_images = np.load("./0220_ordered_232560_imgs.npy")
train_labels = np.load("./0220_ordered_232560_labels.npy")

# train_images = np.load("./128x128_by_lafoss_shuffled.npy")
# train_labels = np.load("./128x128_by_lafoss_shuffled_label.npy")

IMAGE_SIZE = (224,224)
MODEL_TYPE = "101"
USE_CUTMIX = True
USE_MIXUP = False
USE_CUTOUT = False
MOR_AUG_RATE = 0
# CUT_MIX_RATE = 0.9

POST_AUG_RATE = 0

USE_FOCAL_LOSS = False
USE_CLASS_BALANCED_LOSS = False
USE_LABEL_SMOOTHING = False
LS_EPSILON = 0.1

NO_EXTRA_AUG = True

USE_PRETRAINED = True
DROP_RATE = None     ###If no dp, use None instead of 0

USE_AMP = False
OPT_LEVEL = "O2"

USE_MISH = False
FINE_TUNE_EP = 10

FOLD = 0
# stack_model_name = "101_mix_3folds_val1_034"
# stack_model_name = "101_mix_4folds_val0_1234"
# stack_model_name = "101_5folds_mor0.3_val1_034_03"
# stack_model_name = "101_6folds_mor0.3_val1_4_034_03"

# stack_model_name = "101_3folds_mor0.3_CBL_val0_012"
# stack_model_name = "101_3folds_val0_012_TTAMor0.6"
stack_model_name = "101_3folds_mor0.3_CBL_val0_012_v2"


import torch.nn as nn
import torch.nn.functional as F

from se_resnet import *
from se_resnet_mish import se_resnext50_32x4d_mish, Mish
from collections import OrderedDict
def get_resnext_model(model_type="101", pretrained=True, dropout=None):
    if model_type == "101":
        model = se_resnext101_32x4d(num_classes=1000, pretrained=pretrained,dropout=DROP_RATE)
    elif model_type== "50":
        if USE_MISH == True:
            model = se_resnext50_32x4d_mish(num_classes=1000, pretrained=pretrained,dropout=DROP_RATE)
        else:
            model = se_resnext50_32x4d(num_classes=1000, pretrained=pretrained,dropout=DROP_RATE)
    else:
        print("!!!Wrong se_res model structure!!!")
        return
    inplanes = 64  ###inplanes above!!!
    input_channels = 1
    if USE_MISH == True:
        layer0_modules = [
            ('conv1', nn.Conv2d(input_channels, inplanes, kernel_size=7, stride=2,    
                                padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(inplanes)),
            ('mish1', Mish()),
        ]        
    else:
        layer0_modules = [
            ('conv1', nn.Conv2d(input_channels, inplanes, kernel_size=7, stride=2,    
                                padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(inplanes)),
            ('relu1', nn.ReLU(inplace=True)),
        ]           
    layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,ceil_mode=True)))
    model.layer0 = nn.Sequential(OrderedDict(layer0_modules))
    model.classifier_root = nn.Linear(model.feature_dim, 168)
    model.classifier_vowel = nn.Linear(model.feature_dim, 11)
    model.classifier_constant = nn.Linear(model.feature_dim, 7)
    return model


def label_smoothing_criterion(epsilon=0.1, reduction='mean'):
    def _label_smoothing_criterion(preds, targets):
        n_classes = preds.size(1)
        device = preds.device
        onehot = onehot_encoding(targets, n_classes).float().to(device)
        targets = onehot * (1 - epsilon) + torch.ones_like(onehot).to(
            device) * epsilon / n_classes
        loss = cross_entropy_loss(preds, targets, reduction)
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(
                '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

    return _label_smoothing_criterion


###Morphological Augmentation
def trans_morphological(img):
    return func_hybrid(img)

def func_Erosion(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(1, 4, 2)))    
    img = cv2.erode(img, kernel, iterations=1)
    return img

def func_Dilation(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(1, 4, 2)))    
    img = cv2.dilate(img, kernel, iterations=1)
    return img

def get_random_kernel():
    structure = np.random.choice([cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(1, 4, 2)))    
    return kernel

def func_opening(img):
    img = cv2.erode(img, get_random_kernel(), iterations=1)
    img = cv2.dilate(img, get_random_kernel(), iterations=1)
    return img

def func_closing(img):
    img = cv2.dilate(img, get_random_kernel(), iterations=1)
    img = cv2.erode(img, get_random_kernel(), iterations=1)
    return img

def func_hybrid(img):
    rand_tag = np.random.random()
    if rand_tag < 0.25:
        img = func_Erosion(img)
    elif rand_tag < 0.5:
        img = func_Dilation(img)
    elif rand_tag < 0.75:
        img = func_opening(img)
    else:
        img = func_closing(img)        
    return img

trans_norm = transforms.Compose([
        transforms.ToPILImage(),  ###numpy or cv array should be (H,W,C), Tensor should be(C,H,W)
        transforms.ToTensor(),
])



def get_score(preds,targets):
    scores = []
    for i in range(3):
        # print("t shape:",np.shape(targets[i]))
        # print("p shape:",np.shape(preds[i]))
        # print(targets)
        # print(preds)
        scores.append(sklearn.metrics.recall_score(targets[i],preds[i], average='macro'))
    final_score = np.average(scores, weights=[2,1,1])
    return final_score



trans_none = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), #For resnet
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.05302268],std=[0.15688393]) #train_images 128x128 vr 0
        # transforms.Normalize(mean=[0.0530355],std=[0.15949783]) #train_images 224x224 vr 0
])

trans_post = transforms.Compose([
        ###post1
        transforms.ToPILImage(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomAffine(degrees=5,translate=(0.1,0.1),scale=[0.8,1.2],shear=10),

        ###post2
        # transforms.ToPILImage(),
        # transforms.ColorJitter(0.3, 0.3, 0.3),
        # transforms.RandomAffine(degrees=10,translate=(0.2,0.2),scale=[0.7,1.3],shear=10),

        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.05302268],std=[0.15688393]), #train_images 128x128 vr 0
        # transforms.Normalize(mean=[0.0530355],std=[0.15949783]), #train_images 224x224 vr 0 
])

trans_norm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.05302268],std=[0.15688393]), #train_images 128x128 vr 0
        # transforms.Normalize(mean=[0.0530355],std=[0.15949783]), #train_images 224x224 vr 0        
])

trans_val = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), #For resnet
        transforms.ToTensor(),  #Take Image as input and convert to tensor with value from 0 to1  
        # transforms.Normalize(mean=[0.05302268],std=[0.15688393]) #train_images 128x128 vr 0
        # transforms.Normalize(mean=[0.0530355],std=[0.15949783]) #train_images 224x224 vr 0        
    ])

class BengaliDataset(Dataset):
    def __init__(self,data_len=None, is_validate=False,validate_rate=None,indices=None):
        self.is_validate = is_validate
        self.data = train_images
        self.label = train_labels
        if data_len == None:
            data_len = len(self.data)
        
        self.indices = indices
        if self.is_validate:
            self.len = int(data_len*validate_rate)
            self.offset = int(data_len*(1-validate_rate))
            self.transform = trans_val
        else:
            self.len = int(data_len*(1-validate_rate))
            self.offset = 0
            if NO_EXTRA_AUG:
                self.transform = trans_none
            else:
                self.transform = trans

    def set_transform(self,tag):
        self.transform = trans if tag else trans_none

    def __getitem__(self, idx):
        random.seed(np.random.randint(1000000))
        idx += self.offset
        idx = self.indices[idx]
        img = np.uint8(self.data[idx]) #(137,236), value: 0~255
        labels = self.label[idx] #(num,3) grapheme_root, vowel_diacritic, constant_diacritic
        img = Image.fromarray(img)
        img = self.transform(img)     #value: 0~1, shape:(1,137,236)
        label = torch.as_tensor(labels, dtype=torch.uint8)    #value: 0~9, shape(3)
        # aug_tag = False if np.random.random() < CUT_MIX_RATE else True

        return img, labels

    def __len__(self):
        return self.len    

def get_kfold_dataset_loader(k=5,val_rate=0.1,indices_len=None, batch_size=None,num_workers=None):
    train_loader_list = []
    val_loader_list = []
    indices = np.arange(indices_len)
    val_len = indices_len//k
    idx = 0

    for i in range(k):
        ind = np.concatenate([indices[:idx],indices[idx+val_len:],indices[idx:idx+val_len]])
        idx += val_len
        if i!=FOLD:
            continue
        # print("here fold",FOLD,ind[:30],ind[-30:])
        train_dataset = BengaliDataset(data_len=None,is_validate=False, validate_rate=val_rate,indices=ind)
        val_dataset = BengaliDataset(data_len=None,is_validate=True, validate_rate=val_rate, indices=ind)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,worker_init_fn=lambda x: np.random.seed())
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        train_loader_list.append(train_loader)
        val_loader_list.append(val_loader)
        
    return train_loader_list, val_loader_list

def get_model(model_type="50", pretrained=False):
#     model = SE_Net3(in_channels=1)
    # model = EffNet(in_channels=1)
    model = get_resnext_model(model_type=model_type, pretrained=pretrained)
    if device == "cuda":
        model.cuda()
    return model


from sklearn.metrics import mean_squared_error # the metric to test 
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib

# enc_r = OneHotEncoder(sparse=False,categories=[np.arange(168)])
# enc_v = OneHotEncoder(sparse=False,categories=[np.arange(11)])
# enc_c = OneHotEncoder(sparse=False,categories=[np.arange(7)])

def one_hot(target,class_num):
    ###Input:(batch,)
    ###output:(batch,class_num)
    batch_num = len(target)
    one_hot_array = np.zeros(shape=(batch_num,class_num))
    # print(target)
    for i in range(batch_num):
        indice = target[i]   ###label start from 0
        one_hot_array[i][indice] = 1
    return one_hot_array

def save_stack_model(mr,mv,mc):
    # now you can save it to a file
    joblib.dump(mr, 'stack_model_r_{}.pkl'.format(stack_model_name))
    joblib.dump(mv, 'stack_model_v_{}.pkl'.format(stack_model_name))
    joblib.dump(mc, 'stack_model_c_{}.pkl'.format(stack_model_name))
    print("Save stacked model complete")
    return

def load_stack_model():
    # and later you can load it
    mr = joblib.load('stack_model_r_{}.pkl'.format(stack_model_name))
    mv = joblib.load('stack_model_v_{}.pkl'.format(stack_model_name))
    mc = joblib.load('stack_model_c_{}.pkl'.format(stack_model_name))
    print("load stacked model complete")
    return mr,mv,mc

if __name__ == "__main__":
    # x = np.random.random((3,10))
    # # x2 = np.random.random((5,5))
    # # x = np.hstack((x,x2))
    # print(np.shape(x))
    # y = np.random.randint(5,size=(3,))
    # print(y)
    # one_y = one_hot(y,10)
    # print(np.shape(y_onehot))
    # ### fit(x,y)  x:(batch_num,feature_num) 2D,  y:(batch_num,feature,) or (batch_num,feature,one_hot_feature) 
    # ### X needs to be 2D, y needs to be 1D(labels) or 2D(one hot label)
    # meta_model = LinearRegression()
    # meta_model.fit(x,y_onehot)
    # # print("pred:",meta_model.predict(x))
    # stop

    ###Train model
    model_name = []

    ###Stacking model 0313_unseen
    model_name.append("50fold_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vpX_224x224_pre1_Mor0.3uniCutmix1_CBL_noLS_fp16_fold0_Ep160_Fold0_acc99.9433")
    model_name.append("50fold_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vpX_224x224_pre1_Mor0.3uniCutmix1_CBL_noLS_fp16_fold30_Ep168_Fold30_acc99.9265")
    ###v2
    model_name.append("50fold_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vpX_224x224_pre1_Mor0.3uniCutmix1_CBL_noLS_fp16_fold45_Ep183_Fold45_acc99.9653")
    ###v1
    # model_name.append("50fold_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vpX_224x224_pre1_StrongMor0.5uniCutmix1_CBL_noLS_fp16_fold45_Ep180_Fold45_acc99.9680")    

    ###Stacking model 0309
    # model_name.append("5fold_ocp0.15_prcnt20_div70_EP240_Tune10_b128_vp750_280x280_pre1_Fmix0.5Cut0.5_LS_fp16_recall_fold0_Ep250_Fold0_acc99.8885")
    # model_name.append("5fold_ocp0.15_prcnt20_div70_EP240_Tune10_b128_vp750_280x280_pre1_Fmix0.5Cut0.5_LS_fp16_recall_fold3_Ep196_Fold3_acc99.8840")
    # model_name.append("5fold_ocp0.15_prcnt20_div70_EP240_Tune10_b128_vp750_280x280_pre1_Fmix0.5Cut0.5_LS_fp16_recall_fold4_Ep205_Fold4_acc99.8400")
    # model_name.append("5fold_ocp0.15_prcnt20_div70_EP240_Tune10_b128_vp750_280x280_pre1_Mor0.3Fmix0.5Cut0.5_NoLS_fp16_recall_fold4_Ep231_Fold4_acc99.8647")
    # model_name.append("5fold_se101_ocp0.15_prcnt20_div70_EP240_Tune10_b128_vp750_224x224_pre1_Fmix0.5Cut0.5_LS_fp16_recall_fold0_Ep205_Fold0_acc99.8972")
    # model_name.append("5fold_se101_ocp0.15_prcnt20_div70_EP240_Tune10_b128_vp750_224x224_pre1_Fmix0.5Cut0.5_LS_fp16_recall_fold3_Ep223_Fold3_acc99.8770")
    # model_name.append("5fold_se101_ocp0.15_prcnt20_div70_EP240_Tune10_b128_vp750_224x224_pre1_Fmix0.5Cut0.5_LS_fp16_recall_fold4_Ep226_Fold4_acc99.8660")
    # model_name.append("5fold_se101_ocp0.15_prcnt20_div70_EP240_Tune10_b128_vp750_224x224_pre1_Mor0.3Fmix0.5Cut0.5_LS_fp16_recall_fold0_Ep211_Fold0_acc99.8505")
    # model_name.append("5fold_se101_ocp0.15_prcnt20_div70_EP240_Tune10_b128_vp750_224x224_pre1_Mor0.3Fmix0.5Cut0.5_LS_fp16_recall_fold3_Ep231_Fold3_acc99.8341")
    # model_name.append("5fold_se101_ocp0.15_prcnt20_div70_EP240_Tune10_b128_vp750_224x224_pre1_Mor0.3Fmix0.5Cut0.5_LS_fp16_recall_fold4_Ep248_Fold4_acc99.8482")

    ###Stacking model 0305
    # model_name.append("5fold_se101_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vp750_224x224_pre1_Cutmix0.9_LS_fp16_recall_fold0_Ep179_Fold0_acc99.9090")
    # model_name.append("5fold_se101_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vp750_224x224_pre1_Cutmix0.9_LS_fp16_recall_fold1_Ep188_Fold1_acc99.8842")
    # model_name.append("5fold_se101_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vp750_224x224_pre1_Cutmix0.9_LS_fp16_recall_fold2_Ep173_Fold2_acc99.8847")
    # model_name.append("5fold_se101_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vp750_224x224_pre1_Cutmix0.9_LS_fp16_recall_fold3_Ep188_Fold3_acc99.8963")
    # model_name.append("5fold_se101_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vp750_224x224_pre1_Cutmix0.9_LS_fp16_recall_fold4_Ep178_Fold4_acc99.8003")
    #101_mix_5folds_val3_01234v2
    # model_name.append("5fold_se101_ocp0.15_prcnt25_div70_EP150_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold4_Ep159_Fold4_acc99.8506") #recal=99.8102

    ###Stacking model 0304
    # model_name.append("5fold_se101_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold0_Ep176_Fold0_acc99.8855")
    # model_name.append("5fold_se101_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold1_Ep176_Fold1_acc99.8812")
    # model_name.append("5fold_se101_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold2_Ep169_Fold2_acc99.8667")
    # model_name.append("5fold_se101_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold3_Ep181_Fold3_acc99.8699")
    # model_name.append("5fold_se101_ocp0.15_prcnt25_div70_EP150_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold0_Ep146_Fold0_acc99.8608")
    # model_name.append("5fold_se101_ocp0.15_prcnt25_div70_EP150_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold1_Ep155_Fold1_acc99.8495")
    # model_name.append("5fold_se101_ocp0.15_prcnt25_div70_EP150_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold2_Ep145_Fold2_acc99.8603")    

    ###Stacking model 0302
    # model_name.append("5fold_se101_ocp0.15_prcnt25_div70_EP150_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold4_Ep159_Fold4_acc99.8506")
    # model_name.append("5fold_se101_ocp0.15_prcnt25_div70_EP150_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold3_Ep143_Fold3_acc99.8452")
    # model_name.append("5fold_se101_ocp0.15_prcnt25_div70_EP150_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold2_Ep145_Fold2_acc99.8603")    
    # # model_name.append("5fold_se101_ocp0.15_prcnt25_div70_EP150_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold1_Ep155_Fold1_acc99.8495")
    # model_name.append("5fold_se101_ocp0.15_prcnt25_div70_EP150_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold0_Ep146_Fold0_acc99.8608")


    batch_size = 256
    num_workers = 12
    k = 5
    indices_len = 232560
    # indices_len = 200840
    vr = 1/k
    print("validation rate:",vr)
    train_loaders, val_loaders = get_kfold_dataset_loader(k, vr, indices_len, batch_size, num_workers)
    if USE_FOCAL_LOSS == True:
        criterion = FocalLossWithOutOneHot(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("Fold:",FOLD)
    ensemble_root = "./stacking_model_0313_unseen/"
    ensemble_models = []
    data_num = 0
    acc = 0 

    for file_name in model_name:
        if file_name.find("ocp") == -1:
            continue
    
        print(file_name)
        model = get_model(model_type=MODEL_TYPE,pretrained=False)
        if USE_AMP == True:
            model = amp.initialize(model,None,opt_level="O2",keep_batchnorm_fp32=True,verbosity=0,loss_scale="dynamic")
        model.load_state_dict(torch.load("{}/{}".format(ensemble_root,file_name)))
        model.eval()
        ensemble_models.append(model)

    model_num = len(ensemble_models)
    print("len of models:",model_num)    

    for model_i in range(len(ensemble_models)):
        train_loader = train_loaders[0]
        val_loader = val_loaders[0]
        model = ensemble_models[model_i]
        if USE_AMP == True:
            if OPT_LEVEL == "O2":
                model, optimizer = amp.initialize(model, optimizer, opt_level="O2",
                                                                 keep_batchnorm_fp32=True, loss_scale="dynamic")
            elif OPT_LEVEL == "O1":
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            else:
                print("Wrong opt level")

    ###Train meta model
    meta_model_r = LinearRegression()
    meta_model_v = LinearRegression()
    meta_model_c = LinearRegression()

    # meta_model_r = LogisticRegression()
    # meta_model_v = LogisticRegression()
    # meta_model_c = LogisticRegression()

    # meta_model_r = DecisionTreeClassifier()
    # meta_model_v = DecisionTreeClassifier()
    # meta_model_c = DecisionTreeClassifier()

    # meta_model_r = RandomForestClassifier(n_estimators=100, max_features="auto",random_state=0, verbose=True)
    # meta_model_v = RandomForestClassifier(n_estimators=100, max_features="auto",random_state=0, verbose=True)
    # meta_model_c = RandomForestClassifier(n_estimators=100, max_features="auto",random_state=0, verbose=True)

    stack_pred_r = np.empty([0,168*model_num])
    stack_pred_v = np.empty([0,11*model_num])
    stack_pred_c = np.empty([0,7*model_num])
    
    ###For linear regression
    stack_target_r = np.empty([0,168])
    stack_target_v = np.empty([0,11])
    stack_target_c = np.empty([0,7])

    ###For other classification method
    # stack_target_r = np.empty([0,])
    # stack_target_v = np.empty([0,])
    # stack_target_c = np.empty([0,])

    with torch.no_grad():
        for idx, data in enumerate(tqdm(val_loader)):
            img, target = data
            img, target = img.to(device), target.to(device,dtype=torch.long)

            pred_list_root = torch.Tensor([]).to(device)
            pred_list_vow = torch.Tensor([]).to(device)
            pred_list_const = torch.Tensor([]).to(device)

            ###TTA mor stacking
            tmp_rand = np.random.random()
            if tmp_rand < MOR_AUG_RATE:
                for j in range(img.size(0)):
                    tmp_img = trans_morphological(np.uint8(img[j][0].cpu().numpy()*255))  #(h,w)
                    tmp_img = trans_norm(tmp_img)     #(1,h,w)
                    img[j] = tmp_img


            for model_i in range(model_num):
                pred_root, pred_vow, pred_const = ensemble_models[model_i](img) #(batch_num, label_num)
                pred_list_root = torch.cat((pred_list_root,pred_root),dim=1)      #pred_list: (batch_num,168*model_num)
                pred_list_vow = torch.cat((pred_list_vow,pred_vow),dim=1)         #pred_list: (batch_num,11*model_num)
                pred_list_const = torch.cat((pred_list_const,pred_const),dim=1)   #pred_list: (batch_num,7*model_num) 

            tmp_pr = pred_list_root.cpu().numpy()
            tmp_pv = pred_list_vow.cpu().numpy()
            tmp_pc = pred_list_const.cpu().numpy()
            # print("here1",np.shape(tmp_pr))
            # print("here2",np.shape(tmp_pv))
            # print("here3",np.shape(tmp_pc))
            
            stack_pred_r = np.concatenate((stack_pred_r,tmp_pr),axis=0)      ###(total_num,168*model_num)
            stack_pred_v = np.concatenate((stack_pred_v,tmp_pv),axis=0)     ###(total_num,11*model_num)
            stack_pred_c = np.concatenate((stack_pred_c,tmp_pc),axis=0)  ###(total_num,7*model_num)

            tmp_target = target.cpu().numpy()
            tmp_tr, tmp_tv, tmp_tc = tmp_target[:,0],tmp_target[:,1],tmp_target[:,2]   ###(batch,)

            ###For linear regression
            stack_target_r = np.concatenate((stack_target_r,one_hot(tmp_tr,168)),axis=0)   ###(total_num,168)
            stack_target_v = np.concatenate((stack_target_v,one_hot(tmp_tv,11)),axis=0)   ###(total_num,11)
            stack_target_c = np.concatenate((stack_target_c,one_hot(tmp_tc,7)),axis=0)   ###(total_num,7)

            ###For other classification method
            # stack_target_r = np.concatenate((stack_target_r,tmp_tr),axis=0)   ###(total_num,1)
            # stack_target_v = np.concatenate((stack_target_v,tmp_tv),axis=0)   ###(total_num,1)
            # stack_target_c = np.concatenate((stack_target_c,tmp_tc),axis=0)   ###(total_num,1)


        print("here4",np.shape(stack_pred_r))
        print("here5",np.shape(stack_pred_v))
        print("here6",np.shape(stack_pred_c))            
        print("here7",np.shape(stack_target_r))
        print("here8",np.shape(stack_target_v))
        print("here9",np.shape(stack_target_c))            

    ### fit(x,y)  x:(batch_num,feature_num) 2D,  y:(batch_num,feature,) or (batch_num,feature,one_hot_feature) 
    ### X needs to be 2D, y needs to be 1D(labels) or 2D(one hot label)
    print("start linear regression...")
    meta_model_r.fit(stack_pred_r,stack_target_r)
    print("r file completed")
    meta_model_v.fit(stack_pred_v,stack_target_v)
    print("v file completed")
    meta_model_c.fit(stack_pred_c,stack_target_c)
    print("c file completed")
    save_stack_model(meta_model_r,meta_model_v,meta_model_c)
    # print("target:",tmp_tc[:5])
    # print("target one hot:",stack_target_c[:5])
    # pred = meta_model_c.predict(stack_pred_c)
    # print("pred one hot",pred)
    # pred = np.argmax(pred,axis=1)
    # print("pred:",pred[:5])


    ###Inference with meta model:
    # model_name = []

    # # ###Stacking model 0313_unseen
    # model_name.append("50fold_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vpX_224x224_pre1_Mor0.3uniCutmix1_CBL_noLS_fp16_fold0_Ep160_Fold0_acc99.9433")
    # model_name.append("50fold_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vpX_224x224_pre1_Mor0.3uniCutmix1_CBL_noLS_fp16_fold30_Ep168_Fold30_acc99.9265")
    # ###v2
    # model_name.append("50fold_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vpX_224x224_pre1_Mor0.3uniCutmix1_CBL_noLS_fp16_fold45_Ep183_Fold45_acc99.9653")
    # ###v1
    # # model_name.append("50fold_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vpX_224x224_pre1_StrongMor0.5uniCutmix1_CBL_noLS_fp16_fold45_Ep180_Fold45_acc99.9680")

    # ###Stacking model 0302
    # # model_name.append("5fold_se101_ocp0.15_prcnt25_div70_EP150_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold4_Ep159_Fold4_acc99.8506")
    # # model_name.append("5fold_se101_ocp0.15_prcnt25_div70_EP150_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold3_Ep143_Fold3_acc99.8452")
    # # model_name.append("5fold_se101_ocp0.15_prcnt25_div70_EP150_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold2_Ep145_Fold2_acc99.8603")    
    # # # model_name.append("5fold_se101_ocp0.15_prcnt25_div70_EP150_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold1_Ep155_Fold1_acc99.8495")
    # # model_name.append("5fold_se101_ocp0.15_prcnt25_div70_EP150_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold0_Ep146_Fold0_acc99.8608")

    # ###Stacking model 0304
    # # model_name.append("5fold_se101_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold0_Ep176_Fold0_acc99.8855")
    # # model_name.append("5fold_se101_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold2_Ep169_Fold2_acc99.8667")
    # # model_name.append("5fold_se101_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold1_Ep176_Fold1_acc99.8812")
    # # model_name.append("5fold_se101_ocp0.15_prcnt15_div50_EP180_Tune10_b128_vp750_224x224_pre1_way1Mor0.3Cutmix0.9_LS_fp16_fold3_Ep173_Fold3_acc99.8667")
    # # model_name.append("")
    
    # batch_size = 256
    # num_workers = 12
    # k = 5
    # indices_len = 232560
    # vr = 1/k
    # print("validation rate:",vr)
    # train_loaders, val_loaders = get_kfold_dataset_loader(k, vr, indices_len, batch_size, num_workers)
    # if USE_FOCAL_LOSS == True:
    #     criterion = FocalLossWithOutOneHot(gamma=2)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()

    # print("Fold:",FOLD)
    # ensemble_root = "./stacking_model_0313_unseen/"
    # ensemble_models = []
    # data_num = 0
    # acc = 0 

    # for file_name in model_name:
    #     if file_name.find("ocp") == -1:
    #         continue
    #     print(file_name)
    #     model = get_model(model_type=MODEL_TYPE,pretrained=False)
    #     if USE_AMP == True:
    #         model = amp.initialize(model,None,opt_level="O2",keep_batchnorm_fp32=True,verbosity=0,loss_scale="dynamic")
    #     model.load_state_dict(torch.load("{}/{}".format(ensemble_root,file_name)))
    #     model.eval()
    #     ensemble_models.append(model)

    # model_num = len(ensemble_models)
    # print("len of models:",model_num)

    # for model_i in range(len(ensemble_models)):
    #     train_loader = train_loaders[0]
    #     val_loader = val_loaders[0]
    #     model = ensemble_models[model_i]
    #     if USE_AMP == True:
    #         if OPT_LEVEL == "O2":
    #             model, optimizer = amp.initialize(model, optimizer, opt_level="O2",
    #                                                              keep_batchnorm_fp32=True, loss_scale="dynamic")
    #         elif OPT_LEVEL == "O1":
    #             model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    #         else:
    #             print("Wrong opt level")

    # meta_model_r,meta_model_v,meta_model_c = load_stack_model()
    # stack_pred_r = np.empty([0,168*model_num])
    # stack_pred_v = np.empty([0,11*model_num])
    # stack_pred_c = np.empty([0,7*model_num])
    # stack_target_r = np.empty([0,168])
    # stack_target_v = np.empty([0,11])
    # stack_target_c = np.empty([0,7])

    # acc = 0
    # acc_r = 0
    # acc_v = 0
    # acc_c = 0
    # data_num = 0

    # with torch.no_grad():
    #     for idx, data in enumerate(tqdm(val_loader)):
    #         img, target = data
    #         img, target = img.to(device), target.to(device,dtype=torch.long)
    #         pred_list_root = torch.Tensor([]).to(device)
    #         pred_list_vow = torch.Tensor([]).to(device)
    #         pred_list_const = torch.Tensor([]).to(device)
    #         for model_i in range(model_num):
    #             pred_root, pred_vow, pred_const = ensemble_models[model_i](img) #(batch_num, label_num)
    #             pred_list_root = torch.cat((pred_list_root,pred_root),dim=1)      #pred_list: (batch_num,168*model_num)
    #             pred_list_vow = torch.cat((pred_list_vow,pred_vow),dim=1)         #pred_list: (batch_num,11*model_num)
    #             pred_list_const = torch.cat((pred_list_const,pred_const),dim=1)   #pred_list: (batch_num,7*model_num) 

    #         tmp_pr = pred_list_root.cpu().numpy()
    #         tmp_pv = pred_list_vow.cpu().numpy()
    #         tmp_pc = pred_list_const.cpu().numpy()

    #         ###Linear regression            
    #         pred_onehot_r = meta_model_r.predict(tmp_pr)   ###(batch,168)
    #         pred_onehot_v = meta_model_v.predict(tmp_pv)   ###(batch,11)
    #         pred_onehot_c = meta_model_c.predict(tmp_pc)   ###(batch,7)
    #         pred_r_class = np.argmax(pred_onehot_r,axis=1) #(batch,)
    #         pred_v_class = np.argmax(pred_onehot_v,axis=1) #(batch,)
    #         pred_c_class = np.argmax(pred_onehot_c,axis=1) #(batch,)

    #         ###others classfication
    #         # pred_r_class = meta_model_r.predict(tmp_pr)
    #         # pred_v_class = meta_model_v.predict(tmp_pv)
    #         # pred_c_class = meta_model_c.predict(tmp_pc)


    #         tmp_target = target.cpu().numpy()
    #         tmp_tr, tmp_tv, tmp_tc = tmp_target[:,0],tmp_target[:,1],tmp_target[:,2]   ###(batch,class_num)

    #         acc_r += (pred_r_class == tmp_tr).sum()
    #         acc_v += (pred_v_class == tmp_tv).sum()
    #         acc_c += (pred_c_class == tmp_tc).sum()
    #         data_num += img.size(0)

    #     acc_r /= data_num
    #     acc_v /= data_num
    #     acc_c /= data_num
    #     acc += 0.5*acc_r + 0.25*acc_v + 0.25*acc_c
    #     print("acc:{:.4f},accr:{:.4f},accv:{:.4f},accc:{:.4f}".format(acc*100,acc_r*100,acc_v*100,acc_c*100))

    # # fit(x,y)  x:(batch_num,feature_num) 2D,  y:(batch_num,feature,) or (batch_num,feature,one_hot_feature) 
    # # X needs to be 2D, y needs to be 1D(labels) or 2D(one hot label)