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
# import pandas as pd


from torch_lr_finder import LRFinder
from onecyclelr.onecyclelr import OneCycleLR

device = "cuda"

train_images = np.load("./train_images_invert_0204.npy")
train_labels = np.load("./train_labels_shuffle_0204.npy")

IMAGE_SIZE = (128,128)
MODEL_TYPE = "101"
USE_CUTMIX = True
CUT_MIX_RATE = 0.5
FULL_CUTMIX = False
USE_PRETRAINED = True

import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b3')

# out_channels = 1280  #eff-b0
# out_channels = 1536  #eff-b3
out_channels = 2560  #eff-b7

class EffNet(nn.Module):
    def __init__(self,in_channels=3):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b7',in_channels=in_channels)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.2)
        self._fc_root = nn.Linear(out_channels, 168)
        self._fc_vowel = nn.Linear(out_channels, 11)
        self._fc_constant = nn.Linear(out_channels, 7)
    def forward(self, inputs):
        bs = inputs.size(0)
        # Convolution layers
        x = self.backbone.extract_features(inputs)
#         print("feature size:", x.size())
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        out_root = self._fc_root(x)
        out_vowel = self._fc_vowel(x)
        out_constant = self._fc_constant(x)
        return out_root, out_vowel, out_constant
        
        

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
    targets1, targets2,targets3, targets4,targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    criterion2 = nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion2(preds1, targets1) + (1 - lam) * criterion2(preds1, targets2) + lam * criterion2(preds2, targets3) + (1 - lam) * criterion2(preds2, targets4) + lam * criterion2(preds3, targets5) + (1 - lam) * criterion2(preds3, targets6)



trans = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), #For resnet
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.RandomAffine(degrees=10,translate=(0.15,0.15),scale=[0.8,1.2]), #Bengarli baseline
        transforms.ToTensor(),  #Take Image as input and convert to tensor with value from 0 to1  
#         transforms.Normalize(mean=[0.05302372],std=[0.15948994]) #train_images distribution
    ])

trans_none = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), #For resnet
        transforms.ToTensor(),
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
            if FULL_CUTMIX:
                self.transform = trans_none
            else:
                self.transform = trans
        
    def __getitem__(self, idx):
        idx += self.offset
        idx = self.indices[idx]
        
        img = np.uint8(self.data[idx]) #(137,236), value: 0~255
        labels = self.label[idx] #(num,3) grapheme_root, vowel_diacritic, constant_diacritic
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
    model = EffNet(in_channels=1)
    # model = get_resnext_model(model_type=model_type, pretrained=pretrained)
    if device == "cuda":
        model.cuda()
    return model

if __name__ == "__main__":
    epochs = 200
    ensemble_models = []
    lr = 1e-3
    batch_size = 24
    val_period = 5300
    train_period = 100
    num_workers = 12
    k = 1
    indices_len = 200840
    vr = 0.15
    print("validation rate:",vr)
    train_loaders, val_loaders = get_kfold_dataset_loader(k, vr, indices_len, batch_size, num_workers)
    save_file_name = "./Bengali_saved_model/sgd_effb7_b24_vp5300_128x128_pre1_cutmix0.5_lr1e-2_vr_0.15_ocp1e-5_0.16_step20"
    print(save_file_name)

    criterion = torch.nn.CrossEntropyLoss()

    ###LR Finder
    # model = get_model(model_type=MODEL_TYPE,pretrained=USE_PRETRAINED)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-7, momentum=0.9)
    # lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    # trainloader = train_loaders[0]
    # lr_finder.range_test(trainloader, end_lr=100, num_iter=100)
    # lr_finder.plot() # to inspect the loss-learning rate graph
    # lr_finder.reset() # to reset the model and optimizer to their initial state
    # print("Done!")

    ###LR Finder Leslie Smith's approach
    # model = get_model(model_type=MODEL_TYPE,pretrained=USE_PRETRAINED)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-2)
    # trainloader = train_loaders[0]
    # val_loader = val_loaders[0]
    # lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    # lr_finder.range_test(trainloader, val_loader=val_loader, end_lr=1, num_iter=100, step_mode="linear")
    # lr_finder.plot(log_lr=False)
    # lr_finder.reset()
    # print("Done")


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
            # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
            lr_scheduler = OneCycleLR(optimizer, num_steps=20, lr_range=(1e-5,0.16))

#             optimizer = torch.optim.RMSprop(model.parameters(),lr=lr)
            # optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.99))
    #         optimizer = torch.optim.Adagrad(model.parameters(),lr=lr)
    #         optimizer = adabound.AdaBound(model.parameters(), lr=lr, final_lr=0.01,amsbound=True)
            # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    #         lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=period,T_mult=1,eta_min=1e-5) #original 
            # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=30,factor=0.1)

            for ep in range(0,epochs+1):
                model.train()
                for idx, data in enumerate(train_loader):
#                     if idx%10 == 0:
#                         print(idx)
                    img, target = data
                    img, target = img.to(device), target.to(device,dtype=torch.long)

                    cutmix_tag = True if FULL_CUTMIX == True or np.random.random()<CUT_MIX_RATE else False
                    if USE_CUTMIX == True and cutmix_tag == True:
                        img, targets = cutmix(img, target[:,0],target[:,1],target[:,2],alpha=np.random.uniform(0.8,1))
            
                    # pred_root, pred_vowel, pred_constant = model.new_forward(img)
                    pred_root, pred_vowel, pred_constant = model(img)
                    
                    ##Cutmix test
                    if USE_CUTMIX == True and cutmix_tag == True:
                        loss = cutmix_criterion(pred_root,pred_vowel,pred_constant,targets)
#                         print(loss.item())                        
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

                        ###Plateau
                        # lr_scheduler.step(val_loss)               
                        # lr_scheduler.step(-1*acc)
                        
                        ###Others                  
                        lr_scheduler.step()

                        if acc >= max_acc:
                            max_acc = acc
                            min_loss = val_loss
                            best_model_dict = model.state_dict()                    
                            if max_acc>0.977:
                                torch.save(best_model_dict, "{}_Ep{}_Fold{}_acc{:.4f}".format(save_file_name,ep,fold,max_acc*1e2))
                        torch.save(best_model_dict, "{}_Fold{}_current".format(save_file_name,fold))
                        
        #                 if val_loss <= min_loss:
        #                     max_acc = acc
        #                     min_loss = val_loss
        #                     best_model_dict = model.state_dict()

                        print("Val Ep{},Loss:{:.6f},rl{:.4f},vl{:.4f},cl{:.4f},Acc:{:.4f}%,ra:{:.4f}%,va:{:.4f}%,ca:{:.4f}%,lr:{}"
                              .format(ep,val_loss,val_loss_root,val_loss_vowel,val_loss_constant,acc*100,acc_root*100,acc_vowel*100,acc_constant*100,optimizer.param_groups[0]['lr']))

                if optimizer.param_groups[0]['lr'] < 1e-5:
                    break         
                    
            ###K-Fold ensemble: Saved k best model for k dataloader
            print("===================Best Fold:{} Saved Loss:{} Acc:{}==================".format(fold,min_loss,max_acc))
            torch.save(best_model_dict, "{}_Fold{}_loss{:.4f}_acc{:.3f}".format(save_file_name,fold,min_loss*1e3,max_acc*1e2))
            print("======================================================")

            del model
            torch.cuda.empty_cache()
