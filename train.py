import os, time, sys
import numpy as np
# np.random.bit_generator = np.random._bit_generator
import torch
from torchvision import transforms, datasets
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# import pandas as pd
device = "cuda"

train_images = np.load("./train_images_invert_0203.npy")
train_labels = np.load("./train_labels_shuffle_0202.npy")

# train_images = np.load("./128x128_by_lafoss_shuffled.npy")
# train_labels = np.load("./128x128_by_lafoss_shuffled_label.npy")

IMAGE_SIZE = (224,224)
MODEL_TYPE = "50"

USE_CUTMIX = True
CUT_MIX_RATE = 1
USE_MIXUP = False
MIXUP_RATE = 0
GRIDMASK_RATE = 0.3

OHEM_RATE = 0  #only use top 70% loss to do backpropagation

NO_EXTRA_AUG = True
USE_PRETRAINED = True

from se_resnet import *
from collections import OrderedDict
def get_resnext_model(model_type="101", pretrained=True):
    if model_type == "101":
        model = se_resnext101_32x4d(num_classes=1000, pretrained=pretrained)
    elif model_type== "50":
        model = se_resnext50_32x4d(num_classes=1000, pretrained=pretrained)
    else:
        print("!!!Wrong se_res model structure!!!")
        return
    inplanes = 64  ###inplanes above!!!
    input_channels = 1
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



def ohem_loss(cls_pred, cls_target):
    batch_size = cls_pred.size(0) 
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)

    # print("shape here:",ohem_cls_loss.size())  ###([batch])

    keep_num = min(ohem_cls_loss.size()[0], int(batch_size*OHEM_RATE) )

    ###Topk way : torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)
    # ohem_cls_loss, k_indices = torch.topk(ohem_cls_loss, keep_num, largest=False)
    ohem_cls_loss, k_indices = torch.topk(ohem_cls_loss, keep_num, largest=True)
    # print("shape here:",ohem_cls_loss.size())  ###([keep_num])

    ###Origin way
    # if keep_num < sorted_ohem_loss.size()[0]:
    #     sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    #     keep_idx_cuda = idx[:keep_num]
    #     ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]

    # print("shape:",ohem_cls_loss.size())  ###([keep_num])
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss

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
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]
    return data, targets

def cutmix_criterion(preds1,preds2,preds3, targets):
    targets1, targets2,targets3, targets4,targets5, targets6, lam = \
    targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]

    if OHEM_RATE > 0:
        criterion2 = ohem_loss
    else:
        criterion2 = nn.CrossEntropyLoss(reduction='mean')
    
#     print("here", preds1.size(), np.shape(targets1))
    return lam * criterion2(preds1, targets1) + (1 - lam) * \
           criterion2(preds1, targets2) + lam * criterion2(preds2, targets3) + (1 - lam) * \
           criterion2(preds2, targets4) + lam * criterion2(preds3, targets5) + (1 - lam) * criterion2(preds3, targets6)


def mixup(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]

    return data, targets


def mixup_criterion(preds1,preds2,preds3, targets):
    targets1, targets2,targets3, targets4,targets5, targets6, lam = \
    targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]

    if OHEM_RATE > 0:
        criterion2 = ohem_loss
    else:
        criterion2 = nn.CrossEntropyLoss(reduction='mean')

    return lam * criterion2(preds1, targets1) + (1 - lam) * criterion2(preds1, targets2) + \
    lam * criterion2(preds2, targets3) + (1 - lam) * criterion2(preds2, targets4) + \
    lam * criterion2(preds3, targets5) + (1 - lam) * criterion2(preds3, targets6)

import albumentations
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F
class GridMask(DualTransform):
    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                             int(i * grid_h) : int(i * grid_h + grid_h / 2),
                             int(j * grid_w) : int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value
                
                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')



trans = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), #For resnet
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.RandomAffine(degrees=10,translate=(0.15,0.15),scale=[0.8,1.2]), #Bengarli baseline
        transforms.ToTensor(),  #Take Image as input and convert to tensor with value from 0 to1  
#         transforms.Normalize(mean=[0.05302372],std=[0.15948994]) #train_images distribution
    ])

trans_none = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), #For resnet
        transforms.ToTensor(),
])

trans_gridmask = albumentations.Compose([
    GridMask(num_grid=(7,7),rotate=30, p=1.5),
])

trans_val = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), #For resnet
        transforms.ToTensor(),  #Take Image as input and convert to tensor with value from 0 to1  
#         transforms.Normalize(mean=[0.05302372],std=[0.15948994]) #train_images distribution
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
        
    def __getitem__(self, idx):
        idx += self.offset
        idx = self.indices[idx]
        
        img = np.uint8(self.data[idx]) #(137,236), value: 0~255
        labels = self.label[idx] #(num,3) grapheme_root, vowel_diacritic, constant_diacritic
        
        if np.random.random() < GRIDMASK_RATE:
            img = cv2.resize(img, IMAGE_SIZE)
            res = trans_gridmask(image=img)
            img = res['image'].astype(np.float32)
            img = Image.fromarray(img)
            img = trans_none(img)
        else:
            img = Image.fromarray(img)
            img = self.transform(img)     #value: 0~1, shape:(1,137,236)

        label = torch.as_tensor(labels, dtype=torch.uint8)    #value: 0~9, shape(3)
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
        
        train_dataset = BengaliDataset(data_len=None,is_validate=False, validate_rate=val_rate,indices=ind)
        val_dataset = BengaliDataset(data_len=None,is_validate=True, validate_rate=val_rate, indices=ind)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        train_loader_list.append(train_loader)
        val_loader_list.append(val_loader)
        
    return train_loader_list, val_loader_list

def get_model(model_type="50", pretrained=False):
#     model = SE_Net3(in_channels=1)
#     model = EffNet(in_channels=1)
    model = get_resnext_model(model_type=model_type, pretrained=pretrained)
    if device == "cuda":
        model.cuda()
    return model

if __name__ == "__main__":
    epochs = 300
    ensemble_models = []
    lr = 1e-3
    batch_size = 96
    val_period = 1333
    train_period = 100
    num_workers = 12
    k = 1
    indices_len = 200840
    vr = 0.15
    print("validation rate:",vr)
    train_loaders, val_loaders = get_kfold_dataset_loader(k, vr, indices_len, batch_size, num_workers)
    save_file_name = "./B_saved_model_0205/seresnext50_b96_vp1333_224x224_pre1_cutmix1_Gridmask0.3_lr1e-3_vr0.15_cosann_model_train"
    print(save_file_name)

    if OHEM_RATE > 0:
        criterion = ohem_loss
    else:    
        criterion = torch.nn.CrossEntropyLoss()
        
    while True:
        print("Fold:",len(train_loaders))
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
#             optimizer = torch.optim.Adamax(model.parameters(),lr=0.002,weight_decay=0)
    #         optimizer = torch.optim.SGD(model.parameters(),lr=lr)
#             optimizer = torch.optim.RMSprop(model.parameters(),lr=lr)
            optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.99))
    #         optimizer = torch.optim.Adagrad(model.parameters(),lr=lr)
    #         optimizer = adabound.AdaBound(model.parameters(), lr=lr, final_lr=0.01,amsbound=True)
    #         optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=15,T_mult=2,eta_min=1e-5) #original 
            # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15,factor=0.1)
            for ep in range(0,epochs+1):
                model.train()
                for idx, data in enumerate(train_loader):
#                     if idx%10 == 0:
#                         print(idx)
                    img, target = data
                    img, target = img.to(device), target.to(device,dtype=torch.long)


                    ###Cutmix
                    cutmix_tag = True if np.random.random()<CUT_MIX_RATE else False
                    if USE_CUTMIX == True and cutmix_tag == True:
                        img, targets = cutmix(img, target[:,0],target[:,1],target[:,2],alpha=np.random.uniform(0.8,1))
                    
                    ###Mixup
                    # mixup_tag = True if np.random.random()<MIXUP_RATE else False
                    # if USE_MIXUP == True and mixup_tag == True:
                    #     img, targets = mixup(img, target[:,0],target[:,1],target[:,2],alpha=np.random.uniform(0.8,1))
            
                    pred_root, pred_vowel, pred_constant = model.new_forward(img)
                    
                    ##Cutmix test
                    if USE_CUTMIX == True and cutmix_tag == True:
                        loss = cutmix_criterion(pred_root,pred_vowel,pred_constant,targets)
#                         print(loss.item())
                    elif USE_MIXUP == True and mixup_tag == True:
                        loss = mixup_criterion(pred_root,pred_vowel,pred_constant,targets)
                    else:
                        loss_root = criterion(pred_root,target[:,0])
                        loss_vowel = criterion(pred_vowel,target[:,1])
                        loss_constant = criterion(pred_constant,target[:,2])
                        loss = loss_root + loss_vowel + loss_constant
                        loss_root_avg += loss_root.item()
                        loss_vowel_avg += loss_vowel.item()
                        loss_constant_avg += loss_constant.item()
                        loss_avg += loss.item()

                    data_num += img.size(0)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

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
                        with torch.no_grad():
                            for idx, data in enumerate(val_loader):
                                img, target = data
                                img, target = img.to(device), target.to(device,dtype=torch.long)
                                tmp = model(img)
                                pred_root, pred_vowel, pred_constant = model.new_forward(img)
                                
                                val_loss_root += criterion(pred_root, target[:,0]).item()
                                val_loss_vowel += criterion(pred_vowel, target[:,1]).item()
                                val_loss_constant += criterion(pred_constant, target[:,2]).item()

                                # print(pred) 
                                _,pred_class_root = torch.max(pred_root.data, 1)
                                _,pred_class_vowel = torch.max(pred_vowel.data, 1)
                                _,pred_class_constant = torch.max(pred_constant.data, 1)

            #                   print(pred_class)
                                acc_root += (pred_class_root == target[:,0]).sum().item()
                                acc_vowel += (pred_class_vowel == target[:,1]).sum().item()
                                acc_constant += (pred_class_constant == target[:,2]).sum().item()

                                data_num += img.size(0)

                        acc_root /= data_num
                        acc_vowel /= data_num
                        acc_constant /= data_num
                        val_loss_root /= data_num
                        val_loss_vowel /= data_num
                        val_loss_constant /= data_num

                        acc = (2*acc_root + acc_vowel + acc_constant)/4
                        val_loss = (2*val_loss_root + val_loss_vowel + val_loss_constant)/4

#                         ###Plateau
# #                         lr_scheduler.step(val_loss)               
#                         lr_scheduler.step(-1*acc)
                        ###Cosine annealing
                        lr_scheduler.step()   

                        if acc >= max_acc:
                            max_acc = acc
                            min_loss = val_loss
                            best_model_dict = model.state_dict()                    
                            if max_acc>0.98:
                                torch.save(best_model_dict, "{}_Ep{}_Fold{}_acc{:.4f}".format(save_file_name,ep,fold,max_acc*1e2))
                        torch.save(best_model_dict, "{}_Fold{}_current".format(save_file_name,fold))
                        
        #                 if val_loss <= min_loss:
        #                     max_acc = acc
        #                     min_loss = val_loss
        #                     best_model_dict = model.state_dict()

                        print("Val Ep{},Loss:{:.6f},rl{:.4f},vl{:.4f},cl{:.4f},Acc:{:.4f}%,ra:{:.4f}%,va:{:.4f}%,ca:{:.4f}%,lr:{}"
                              .format(ep,val_loss,val_loss_root,val_loss_vowel,val_loss_constant,acc*100,acc_root*100,acc_vowel*100,acc_constant*100,optimizer.param_groups[0]['lr']))
                        
                        ##Don't forget change model back to train()
                        model.train()

                ###Plateau
                # if optimizer.param_groups[0]['lr'] < 1e-6:
                #     break         
                    
            ###K-Fold ensemble: Saved k best model for k dataloader
            print("===================Best Fold:{} Saved Loss:{} Acc:{}==================".format(fold,min_loss,max_acc))
            torch.save(best_model_dict, "{}_Fold{}_loss{:.4f}_acc{:.3f}".format(save_file_name,fold,min_loss*1e3,max_acc*1e2))
            print("======================================================")

            del model
            torch.cuda.empty_cache()
