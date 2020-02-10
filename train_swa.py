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
from torch_lr_finder import LRFinder
from One_Cycle_Policy import OneCycle
# from class_balanced_loss import get_cb_loss
from apex import amp

from swa import utils

device = "cuda"

train_images = np.load("./train_images_invert_0204.npy")
train_labels = np.load("./train_labels_shuffle_0204.npy")

# train_images = np.load("./128x128_by_lafoss_shuffled.npy")
# train_labels = np.load("./128x128_by_lafoss_shuffled_label.npy")

IMAGE_SIZE = (128,128)
MODEL_TYPE = "50"
USE_CUTMIX = True
CUT_MIX_RATE = 1

USE_FOCAL_LOSS = False
USE_CLASS_BALANCED_LOSS = False
NO_EXTRA_AUG = True
USE_PRETRAINED = True

USE_AMP = False
USE_MISH = False


###SWA
lr_init = 0.1
wd = 5e-4
swa = True 
swa_start = 100
swa_lr = 0.05
swa_c_epochs = 1


import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b3')

# out_channels = 1280  #eff-b0
# out_channels = 1536  #eff-b3
# out_channels = 2560  #eff-b7


from se_resnet import *
from se_resnet_mish import se_resnext50_32x4d_mish, Mish
from collections import OrderedDict
def get_resnext_model(model_type="101", pretrained=True):
    if model_type == "101":
        model = se_resnext101_32x4d(num_classes=1000, pretrained=pretrained)
    elif model_type== "50":
        if USE_MISH == True:
            model = se_resnext50_32x4d_mish(num_classes=1000, pretrained=pretrained)
        else:
            model = se_resnext50_32x4d(num_classes=1000, pretrained=pretrained)
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
    
    if USE_CLASS_BALANCED_LOSS == True:
        criterion2 = get_cb_loss
    elif USE_FOCAL_LOSS == True:
        criterion2 = FocalLossWithOutOneHot(gamma=2)
    else:
        criterion2 = nn.CrossEntropyLoss(reduction='mean')

    # tmp_loss = criterion2(preds3, targets5)
    # print("here",tmp_loss)

#     print("here", preds1.size(), np.shape(targets1))
    return lam * criterion2(preds1, targets1) + (1 - lam) * \
           criterion2(preds1, targets2) + lam * criterion2(preds2, targets3) + (1 - lam) * \
           criterion2(preds2, targets4) + lam * criterion2(preds3, targets5) + (1 - lam) * criterion2(preds3, targets6)

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


trans = transforms.Compose([
        ###Weak1
        transforms.Resize(IMAGE_SIZE),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.RandomAffine(degrees=3,translate=(0.05,0.05),scale=[0.95,1.05],shear=3),
        transforms.ToTensor(),  #Take Image as input and convert to tensor with value from 0 to1  
        transforms.Normalize(mean=[0.05302268],std=[0.15688393]) #train_images 128x128 vr 0
        # transforms.Normalize(mean=[0.0530355],std=[[0.15949783]]) #train_images 224x224 vr 0
    ])

trans_none = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), #For resnet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.05302268],std=[0.15688393]) #train_images 128x128 vr 0

])

trans_val = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), #For resnet
        transforms.ToTensor(),  #Take Image as input and convert to tensor with value from 0 to1  
        transforms.Normalize(mean=[0.05302268],std=[0.15688393]) #train_images 128x128 vr 0
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
    # model = EffNet(in_channels=1)
    model = get_resnext_model(model_type=model_type, pretrained=pretrained)
    if device == "cuda":
        model.cuda()
    return model


def swa_schedule(epoch):
    t = (epoch) / (swa_start if swa else epochs)
    lr_ratio = swa_lr / lr_init if swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_init * factor


if __name__ == "__main__":
    epochs = 200
    ensemble_models = []
    lr = lr_init
    batch_size = 96
    val_period = 1333
    train_period = 1
    num_workers = 12
    k = 1
    indices_len = 200840
    vr = 0.05
    print("validation rate:",vr)
    train_loaders, val_loaders = get_kfold_dataset_loader(k, vr, indices_len, batch_size, num_workers)
    save_file_name = "./B_saved_model_0208/ocp0.15_prcnt10_div50_EP160_b256_vp500_128x128_pre1_cutmix1_noExtraAug_norm_vr0.15"
    print(save_file_name)

    if USE_FOCAL_LOSS == True:
        criterion = FocalLossWithOutOneHot(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("Fold:",len(train_loaders))
    for fold in range(0,len(train_loaders)):
        train_loader = train_loaders[fold]
        val_loader = val_loaders[fold]

        ###Calculate mean and std
        # mean, std = get_dataset_mean_std(train_loader)
        # print("Average mean:",mean)
        # print("Average std:", std)

        model = get_model(model_type=MODEL_TYPE,pretrained=USE_PRETRAINED)
        max_acc = 0
        min_loss = 10000
        best_model_dict = None
        data_num = 0
        loss_avg = 0
        loss_root_avg = 0
        loss_vowel_avg = 0
        loss_constant_avg = 0
        swa_ep = 0
        swa_n = 0
#       optimizer = torch.optim.Adamax(model.parameters(),lr=0.002,weight_decay=0)
#       optimizer = torch.optim.SGD(model.parameters(),lr=lr)
        
        ###For swa
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_init, momentum=0.9, weight_decay=wd)

        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.95, weight_decay=1e-4)
        # cycle_len = indices_len*(1-vr)*epochs/batch_size
        # onecycle = OneCycle.OneCycle(cycle_len, 0.15, prcnt=10, div=50, momentum_vals=(0.95, 0.8))

        ###SWA
        swa_model = get_model(model_type=MODEL_TYPE,pretrained=USE_PRETRAINED)
        swa_model.cuda()
        swa_n = 0


        if USE_AMP == True:
            model, optimizer = model, optimizer = amp.initialize(model, optimizer, opt_level="O2",
                                                                 keep_batchnorm_fp32=True, loss_scale="dynamic")

        for ep in range(0,epochs+1):
            model.train()
            for idx, data in enumerate(train_loader):
                ###Onecycle policy
                # lr, mom = onecycle.calc()
                # for g in optimizer.param_groups:
                #     g['lr'] = lr
                # for g in optimizer.param_groups:
                #     g['momentum'] = mom
                
                img, target = data
                img, target = img.to(device), target.to(device,dtype=torch.long)

                swa_lr = swa_schedule(ep)
                utils.adjust_learning_rate(optimizer, swa_lr)


                cutmix_tag = True if np.random.random()<CUT_MIX_RATE else False
                if USE_CUTMIX == True and cutmix_tag == True:
                    img, targets = cutmix(img, target[:,0],target[:,1],target[:,2],alpha=np.random.uniform(0.8,1))
        
                # pred_root, pred_vowel, pred_constant = model.new_forward(img)
                pred_root, pred_vowel, pred_constant = model(img)
                
                ##Cutmix test
                if USE_CUTMIX == True and cutmix_tag == True:
                    loss = cutmix_criterion(pred_root,pred_vowel,pred_constant,targets)
                    # print(loss.item())                        
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

                    ###SWA update
                    r_preds_swa = []
                    v_preds_swa = []
                    c_preds_swa = []
                    swa_ep += 1
                    if swa and (swa_ep + 1) >= swa_start and (swa_ep + 1 - swa_start) % swa_c_epochs == 0:
                        utils.moving_average(swa_model, model, 1.0 / (swa_n + 1))
                        swa_n += 1
                        utils.bn_update(train_loader, swa_model)
                        pred_root, pred_vowel, pred_constant = model(img)

                    with torch.no_grad():
                        for idx, data in enumerate(val_loader):
                            img, target = data
                            img, target = img.to(device), target.to(device,dtype=torch.long)
                            tmp = model(img)

                            pred_root, pred_vowel, pred_constant = model(img)
                            pred_root_swa, pred_vowel_swa, pred_constant_swa = model(img)
                            
                            val_loss_root += criterion(pred_root, target[:,0]).item()
                            val_loss_vowel += criterion(pred_vowel, target[:,1]).item()
                            val_loss_constant += criterion(pred_constant, target[:,2]).item()

                            # print(pred) 
                            _,pred_class_root = torch.max(pred_root.data, 1)
                            _,pred_class_vowel = torch.max(pred_vowel.data, 1)
                            _,pred_class_constant = torch.max(pred_constant.data, 1)


                            _,pred_class_root_swa = torch.max(pred_root_swa.data, 1)
                            _,pred_class_vowel_swa = torch.max(pred_vowel_swa.data, 1)
                            _,pred_class_constant_swa = torch.max(pred_constant_swa.data, 1)


                            ###Origin metric
                            # acc_root += (pred_class_root == target[:,0]).sum().item()
                            # acc_vowel += (pred_class_vowel == target[:,1]).sum().item()
                            # acc_constant += (pred_class_constant == target[:,2]).sum().item()
                            # data_num += img.size(0)

                            ###Contest metric
                            r_preds.append(pred_class_root.cpu().numpy())
                            v_preds.append(pred_class_vowel.cpu().numpy())
                            c_preds.append(pred_class_constant.cpu().numpy())

                            r_preds_swa.append(pred_class_root_swa.cpu().numpy())
                            v_preds_swa.append(pred_class_vowel_swa.cpu().numpy())
                            c_preds_swa.append(pred_class_constant_swa.cpu().numpy())

                            r_targets.append(target[:,0].cpu().numpy())
                            v_targets.append(target[:,1].cpu().numpy())
                            c_targets.append(target[:,2].cpu().numpy())


                    ####Contest metric
                    r_preds = [tmp_j for tmp_i in r_preds for tmp_j in tmp_i]
                    v_preds = [tmp_j for tmp_i in v_preds for tmp_j in tmp_i]
                    c_preds = [tmp_j for tmp_i in c_preds for tmp_j in tmp_i]

                    r_preds_swa = [tmp_j for tmp_i in r_preds_swa for tmp_j in tmp_i]
                    v_preds_swa = [tmp_j for tmp_i in v_preds_swa for tmp_j in tmp_i]
                    c_preds_swa = [tmp_j for tmp_i in c_preds_swa for tmp_j in tmp_i]

                    r_targets = [tmp_j for tmp_i in r_targets for tmp_j in tmp_i]
                    v_targets = [tmp_j for tmp_i in v_targets for tmp_j in tmp_i]
                    c_targets = [tmp_j for tmp_i in c_targets for tmp_j in tmp_i]
                    acc = get_score([r_preds,v_preds,c_preds],[r_targets,v_targets,c_targets])
                    acc_swa = get_score([r_preds_swa,v_preds_swa,c_preds_swa],[r_targets,v_targets,c_targets])

                    # print("final score:",acc)

                    ###Plateau
                    # lr_scheduler.step(val_loss)               
                    # lr_scheduler.step(-1*acc)
                    
                    ###Others                  
                    # lr_scheduler.step()

                    ###Origin metric
                    # print("Val Ep{},Loss:{:.6f},rl{:.4f},vl{:.4f},cl{:.4f},Acc:{:.4f}%,ra:{:.4f}%,va:{:.4f}%,ca:{:.4f}%,lr:{}"
                    #         .format(ep,val_loss,val_loss_root,val_loss_vowel,val_loss_constant,acc*100,acc_root*100,acc_vowel*100,acc_constant*100,optimizer.param_groups[0]['lr']))

                    ###Contest metric
                    print("Val Ep{},Loss:{:.6f},rl{:.4f},vl{:.4f},cl{:.4f},Acc:{:.4f}%,SWA,Acc:{:.4f}%, lr:{}"
                            .format(ep,val_loss,val_loss_root,val_loss_vowel,val_loss_constant,acc*100,acc_swa*100,optimizer.param_groups[0]['lr']))

                    if acc_swa >= max_acc:
                        max_acc = acc_swa
                        min_loss = val_loss
                        best_model_dict = model.state_dict()                    
                        if max_acc>0.98:
                            torch.save(best_model_dict, "{}_Ep{}_Fold{}_acc_swa{:.4f}".format(save_file_name,ep,fold,max_acc*1e2))
                    torch.save(best_model_dict, "{}_Fold{}_current".format(save_file_name,fold))
                    
    #                 if val_loss <= min_loss:
    #                     max_acc = acc
    #                     min_loss = val_loss
    #                     best_model_dict = model.state_dict()

                    ##Don't forget change model back to train()
                    model.train()
                    
            
            
            # if optimizer.param_groups[0]['lr'] < 1e-5:
            #     break         
                
        ###K-Fold ensemble: Saved k best model for k dataloader
        print("===================Best Fold:{} Saved Loss:{} Acc:{}==================".format(fold,min_loss,max_acc))
        torch.save(best_model_dict, "{}_Fold{}_loss{:.4f}_acc{:.3f}".format(save_file_name,fold,min_loss*1e3,max_acc*1e2))
        print("======================================================")

        del model
        torch.cuda.empty_cache()
