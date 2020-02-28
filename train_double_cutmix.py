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

from torch_lr_finder import LRFinder
from One_Cycle_Policy import OneCycle
from class_balanced_loss import get_cb_loss
from apex import amp
import cv2

device = "cuda"

train_images = np.load("./0220_ordered_232560_imgs.npy")
train_labels = np.load("./0220_ordered_232560_labels.npy")

IMAGE_SIZE = (224,224)
MODEL_TYPE = "50"
USE_CUTMIX = True
USE_MIXUP = False
USE_CUTOUT = False
MOR_AUG_RATE = 0.3
CUT_MIX_RATE = 1

DOUBLE_CUT_RATE = 0.6

POST_AUG_RATE = 0

USE_FOCAL_LOSS = False
USE_CLASS_BALANCED_LOSS = False
USE_LABEL_SMOOTHING = True
LS_EPSILON = 0.1

NO_EXTRA_AUG = True

USE_PRETRAINED = True
DROP_RATE = None     ###If no dp, use None instead of 0

USE_AMP = True
OPT_LEVEL = "O2"

USE_MISH = False

FINE_TUNE_EP = 10

FOLD = 3


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


def get_dataset_mean_std(dataloader):
    print("Calculate distribution:")
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in dataloader:
        img = data[0].to(device)
        batch_samples = img.size(0)
        img = img.contiguous().view(batch_samples, img.size(1), -1)
        mean += img.mean(2).sum(0)
        std += img.std(2).sum(0)
        nb_samples += batch_samples
        if nb_samples%5120 == 0:
            print("Finished:", nb_samples)
    print("num of samples:",nb_samples)
    mean /= nb_samples
    std /= nb_samples
    # print("Average mean:",mean)
    # print("Average std:", std)
    return mean.cpu().numpy(), std.cpu().numpy()

def get_augmented_img(img, func):
    output_img = np.zeros((h * 2, w), dtype=np.uint8)
    output_img[:h] = img
    output_img[h:] = func(img)
    return output_img


class FocalLossWithOutOneHot(nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLossWithOutOneHot, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
        logit = logit.clamp(self.eps, 1. - self.eps)
        logit_ls = torch.log(logit)
        loss = F.nll_loss(logit_ls, target, reduction="none")
        view = target.size() + (1,)
        index = target.view(*view)
        loss = loss * (1 - logit.gather(1, index).squeeze(1)) ** self.gamma # focal loss
        return 0.01*loss.sum()

def onehot_encoding(label, n_classes):
    return torch.zeros(label.size(0), n_classes).to(label.device).scatter_(
        1, label.view(-1, 1), 1)

def cross_entropy_loss(input, target, reduction):
    logp = F.log_softmax(input, dim=1)
    loss = torch.sum(-logp * target, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

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

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2
    
def cutmix(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    # shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]
    return data, targets

def cutmix_criterion(preds1,preds2,preds3, targets):
    targets1, targets2,targets3, targets4,targets5, targets6, lam = \
    targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    
    if USE_CLASS_BALANCED_LOSS == True:
        criterion2 = get_cb_loss
    elif USE_FOCAL_LOSS == True:
        criterion2 = FocalLossWithOutOneHot(gamma=2)
    elif USE_LABEL_SMOOTHING == True:
        criterion2 = label_smoothing_criterion(epsilon=LS_EPSILON)        
    else:
        criterion2 = nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion2(preds1, targets1) + (1 - lam) * \
           criterion2(preds1, targets2) + lam * criterion2(preds2, targets3) + (1 - lam) * \
           criterion2(preds2, targets4) + lam * criterion2(preds3, targets5) + (1 - lam) * criterion2(preds3, targets6)

def double_cutmix(data, tr, tv, tc, tr2, tv2, tc2, lam,alpha2):
    indices = torch.randperm(data.size(0))
    s_tr = tr[indices]
    s_tv = tv[indices]
    s_tc = tc[indices]

    s_tr2 = tr2[indices]
    s_tv2 = tv2[indices]
    s_tc2 = tc2[indices]

    lam2 = np.random.beta(alpha2, alpha2)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam2)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    lam2 = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = [tr,tr2,s_tr,s_tr2,tv,tv2,s_tv,s_tv2,tc,tc2,s_tc,s_tc2,lam,lam2]
    return data, targets

def double_cutmix_criterion(preds1,preds2,preds3,tar):
    tr,tr2,s_tr,s_tr2,tv,tv2,s_tv,s_tv2,tc,tc2,s_tc,s_tc2,lam,lam2 = \
    tar[0],tar[1],tar[2],tar[3],tar[4],tar[5],tar[6],tar[7],tar[8],tar[9],tar[10],tar[11],tar[12],tar[13]

    if USE_CLASS_BALANCED_LOSS == True:
        criterion2 = get_cb_loss
    elif USE_FOCAL_LOSS == True:
        criterion2 = FocalLossWithOutOneHot(gamma=2)
    elif USE_LABEL_SMOOTHING == True:
        criterion2 = label_smoothing_criterion(epsilon=LS_EPSILON)        
    else:
        criterion2 = nn.CrossEntropyLoss(reduction='mean')

    return  lam2*lam*criterion2(preds1,tr)+lam2*(1-lam)*criterion2(preds1,s_tr)+\
            (1-lam2)*lam*criterion2(preds1,tr2)+(1-lam2)*(1-lam)*criterion2(preds1,s_tr2)+\
            lam2*lam*criterion2(preds2,tv)+lam2*(1-lam)*criterion2(preds2,s_tv)+\
            (1-lam2)*lam*criterion2(preds2,tv2)+(1-lam2)*(1-lam)*criterion2(preds2,s_tv2)+\
            lam2*lam*criterion2(preds3,tc)+lam2*(1-lam)*criterion2(preds3,s_tc)+\
            (1-lam2)*lam*criterion2(preds3,tc2)+(1-lam2)*(1-lam)*criterion2(preds3,s_tc2)


def cutout(data, targets1, targets2, targets3, alpha):
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = 0
    targets = [targets1, targets2, targets3]
    return data, targets

def cutout_criterion(preds1,preds2,preds3, targets):
    targets1, targets2,targets3  = targets[0], targets[1], targets[2]
    if USE_CLASS_BALANCED_LOSS == True:
        criterion2 = get_cb_loss
    elif USE_FOCAL_LOSS == True:
        criterion2 = FocalLossWithOutOneHot(gamma=2)
    elif USE_LABEL_SMOOTHING == True:
        criterion2 = label_smoothing_criterion(epsilon=LS_EPSILON)        
    else:
        criterion2 = nn.CrossEntropyLoss(reduction='mean')
    return criterion2(preds1, targets1) + criterion2(preds2, targets2) + criterion2(preds3, targets3)



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


###Morphological Augmentation
def trans_morphological(img):
    return func_hybrid(img)

def func_Erosion(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(1, 5, 2)))
    img = cv2.erode(img, kernel, iterations=1)
    return img

def func_Dilation(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(1, 5, 2)))
    img = cv2.dilate(img, kernel, iterations=1)
    return img

def get_random_kernel():
    structure = np.random.choice([cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS])
    kernel = cv2.getStructuringElement(structure, tuple(np.random.randint(1, 5, 2)))
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

if __name__ == "__main__":
    epochs = 180
    ensemble_models = []
    lr = 1e-5
    batch_size = 200
    val_period = 640
    train_period = 1
    num_workers = 12
    k = 7
    indices_len = 232560
    vr = 1/k
    print("validation rate:",vr)
    train_loaders, val_loaders = get_kfold_dataset_loader(k, vr, indices_len, batch_size, num_workers)
    save_file_name = "./B_saved_model_0228_dCut/7fold_ocp0.15_prcnt15_div50_EP180_Tune10_b200_vp640_224x224_pre1_mor0.3Cutmix1Double0.6_LS_fp16_fold{}".format(FOLD)
    print(save_file_name)

    if USE_FOCAL_LOSS == True:
        criterion = FocalLossWithOutOneHot(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("Fold:",FOLD)

    for fold in range(0,len(train_loaders)):
        train_loader = train_loaders[fold]
        val_loader = val_loaders[fold]

        model = get_model(model_type=MODEL_TYPE,pretrained=USE_PRETRAINED)
        max_acc = 0
        min_loss = 10000
        best_model_dict = None
        data_num = 0
        loss_avg = 0
        loss_root_avg = 0
        loss_vowel_avg = 0
        loss_constant_avg = 0

        best_lr = 0
        best_mom = 0

        cutmix_tag = True
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.95, weight_decay=1e-4)
        cycle_len = indices_len*(1-vr)*epochs/batch_size
        onecycle = OneCycle.OneCycle(cycle_len, 0.15, prcnt=15, div=50, momentum_vals=(0.95, 0.8))

        if USE_AMP == True:
            if OPT_LEVEL == "O2":
                model, optimizer = amp.initialize(model, optimizer, opt_level="O2",
                                                                 keep_batchnorm_fp32=True, loss_scale="dynamic")
            elif OPT_LEVEL == "O1":
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            else:
                print("Wrong opt level")

        for ep in range(0,epochs+1+FINE_TUNE_EP):
            model.train()
            for idx, data in enumerate(train_loader):
                ###Onecycle policy
                if ep <epochs:
                    lr, mom = onecycle.calc()
                ###Fine tune 
                else:
                    lr, mom = best_lr, best_mom

                for g in optimizer.param_groups:
                    g['lr'] = lr
                for g in optimizer.param_groups:
                    g['momentum'] = mom

                img, target = data
                img, target = img.to(device), target.to(device,dtype=torch.long)
                
                tmp_rand = np.random.random()
                if tmp_rand < MOR_AUG_RATE:
                    ###Morphological aug
                    for j in range(img.size(0)):
                        tmp_img = trans_morphological(np.uint8(img[j][0].cpu().numpy()*255))
                        tmp_img = trans_norm(tmp_img)     #(1,h,w)
                        img[j] = tmp_img

                ###Cutmix
                tmp_rand = np.random.random()
                cutmix_tag = True if cutmix_tag and tmp_rand<CUT_MIX_RATE else False
                if USE_CUTMIX == True and cutmix_tag == True:
                    img, targets = cutmix(img, target[:,0],target[:,1],target[:,2],alpha=np.random.uniform(0.8,1))
                elif USE_MIXUP:
                    img, targets = mixup(img, target[:,0],target[:,1],target[:,2],alpha=np.random.uniform(0.6,1))
                elif USE_CUTOUT:
                    img, targets = cutout(img, target[:,0],target[:,1],target[:,2],alpha=np.random.uniform(0.6,1))

                ###Double Cutmix
                tmp_rand = np.random.random()
                double_cutmix_tag = True if tmp_rand<DOUBLE_CUT_RATE else False
                if double_cutmix_tag:
                    tr,tr2,tv,tv2,tc,tc2,lam = targets[0],targets[1],targets[2],targets[3],targets[4],targets[5],targets[6]                    
                    img, targets = double_cutmix(img, tr, tv, tc, tr2, tv2, tc2, lam,alpha2=np.random.uniform(0.8,1))
                pred_root, pred_vowel, pred_constant = model(img)

                if double_cutmix_tag:
                    loss = double_cutmix_criterion(pred_root,pred_vowel,pred_constant,targets)
                elif USE_CUTMIX == True and cutmix_tag == True:
                    loss = cutmix_criterion(pred_root,pred_vowel,pred_constant,targets)
                elif USE_MIXUP == True:
                    loss = mixup_criterion(pred_root,pred_vowel,pred_constant,targets)
                elif USE_CUTOUT == True:
                    loss = cutout_criterion(pred_root,pred_vowel,pred_constant,targets)
                else:
                    loss_root = criterion(pred_root,target[:,0])
                    loss_vowel = criterion(pred_vowel,target[:,1])
                    loss_constant = criterion(pred_constant,target[:,2])
                    loss = loss_root + loss_vowel + loss_constant
                    loss_root_avg += loss_root.item()
                    loss_vowel_avg += loss_vowel.item()
                    loss_constant_avg += loss_constant.item()
                    loss_avg += loss.item()
                
                # print(loss.item())

                data_num += img.size(0)
                optimizer.zero_grad()

                if USE_AMP == True:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()

                ###Cosine annealing
    #             lr_scheduler.step()                    

                ###Validation
                if idx!=0 and idx%val_period == 0:
                    model.eval()
                    acc_root = 0
                    acc_vowel = 0
                    acc_constant = 0
                    acc = 0
                    val_loss_root = 0
                    val_loss_vowel = 0
                    val_loss_constant = 0
                    val_loss = 0
                    data_num  = 0

                    r_preds = []
                    v_preds = []
                    c_preds = []
                    r_targets = []
                    v_targets = []
                    c_targets = []

                    with torch.no_grad():
                        for idx, data in enumerate(val_loader):
                            img, target = data
                            img, target = img.to(device), target.to(device,dtype=torch.long)
                            tmp = model(img)
                            # pred_root, pred_vowel, pred_constant = model.new_forward(img)
                            pred_root, pred_vowel, pred_constant = model(img)
                            
                            val_loss_root += criterion(pred_root, target[:,0]).item()
                            val_loss_vowel += criterion(pred_vowel, target[:,1]).item()
                            val_loss_constant += criterion(pred_constant, target[:,2]).item()

                            # print(pred) 
                            _,pred_class_root = torch.max(pred_root.data, 1)
                            _,pred_class_vowel = torch.max(pred_vowel.data, 1)
                            _,pred_class_constant = torch.max(pred_constant.data, 1)

                            ###Origin metric
                            acc_root += (pred_class_root == target[:,0]).sum().item()
                            acc_vowel += (pred_class_vowel == target[:,1]).sum().item()
                            acc_constant += (pred_class_constant == target[:,2]).sum().item()
                            data_num += img.size(0)

                            ###Contest metric
                            # r_preds.append(pred_class_root.cpu().numpy())
                            # v_preds.append(pred_class_vowel.cpu().numpy())
                            # c_preds.append(pred_class_constant.cpu().numpy())
                            # r_targets.append(target[:,0].cpu().numpy())
                            # v_targets.append(target[:,1].cpu().numpy())
                            # c_targets.append(target[:,2].cpu().numpy())

                    ###Origin metric
                    acc_root /= data_num
                    acc_vowel /= data_num
                    acc_constant /= data_num
                    val_loss_root /= data_num
                    val_loss_vowel /= data_num
                    val_loss_constant /= data_num
                    acc = (2*acc_root + acc_vowel + acc_constant)/4
                    val_loss = (2*val_loss_root + val_loss_vowel + val_loss_constant)/4

                    ####Contest metric
                    # r_preds = [tmp_j for tmp_i in r_preds for tmp_j in tmp_i]
                    # v_preds = [tmp_j for tmp_i in v_preds for tmp_j in tmp_i]
                    # c_preds = [tmp_j for tmp_i in c_preds for tmp_j in tmp_i]
                    # r_targets = [tmp_j for tmp_i in r_targets for tmp_j in tmp_i]
                    # v_targets = [tmp_j for tmp_i in v_targets for tmp_j in tmp_i]
                    # c_targets = [tmp_j for tmp_i in c_targets for tmp_j in tmp_i]
                    # acc = get_score([r_preds,v_preds,c_preds],[r_targets,v_targets,c_targets])
                    # print("final score:",acc)

                    ###Plateau
                    # lr_scheduler.step(val_loss)               
                    # lr_scheduler.step(-1*acc)
                    
                    ###Others                  
                    # lr_scheduler.step()

                    ###Origin metric
                    print("Val Ep{},Loss:{:.6f},rl{:.4f},vl{:.4f},cl{:.4f},Acc:{:.4f}%,ra:{:.4f}%,va:{:.4f}%,ca:{:.4f}%,lr:{}"
                            .format(ep,val_loss,val_loss_root,val_loss_vowel,val_loss_constant,acc*100,acc_root*100,acc_vowel*100,acc_constant*100,optimizer.param_groups[0]['lr']))

                    ###Contest metric
                    # print("Val Ep{},Loss:{:.6f},rl{:.4f},vl{:.4f},cl{:.4f},Acc:{:.4f}%,lr:{}"
                    #         .format(ep,val_loss,val_loss_root,val_loss_vowel,val_loss_constant,acc*100,optimizer.param_groups[0]['lr']))

                    if acc >= max_acc:
                        max_acc = acc
                        min_loss = val_loss
                        best_model_dict = model.state_dict()                    
                        best_lr = optimizer.param_groups[0]['lr']
                        best_mom = optimizer.param_groups[0]['momentum']
                        if max_acc>0.998:
                            torch.save(best_model_dict, "{}_Ep{}_Fold{}_acc{:.4f}".format(save_file_name,ep,FOLD,max_acc*1e2))
                    torch.save(best_model_dict, "{}_Fold{}_current".format(save_file_name,FOLD))
                    
                    ##Don't forget change model back to train()
                    model.train()

            # if optimizer.param_groups[0]['lr'] < 1e-5:
            #     break         
                
        ###K-Fold ensemble: Saved k best model for k dataloader
        print("===================Best Fold:{} Saved Loss:{} Acc:{}==================".format(FOLD,min_loss,max_acc))
        torch.save(best_model_dict, "{}_Fold{}_loss{:.4f}_acc{:.3f}".format(save_file_name,FOLD,min_loss*1e3,max_acc*1e2))
        print("======================================================")

        del model
        torch.cuda.empty_cache()
