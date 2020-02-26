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

# train_images = np.load("./train_images_invert_0203.npy")
# train_labels = np.load("./train_labels_shuffle_0202.npy")

train_images = np.load("./0220_ordered_232560_imgs.npy")
train_labels = np.load("./0220_ordered_232560_labels.npy")

# train_images = np.load("./128x128_by_lafoss_shuffled.npy")
# train_labels = np.load("./128x128_by_lafoss_shuffled_label.npy")

IMAGE_SIZE = (224,224)
MODEL_TYPE = "50"
USE_CUTMIX = True
USE_MIXUP = False
USE_CUTOUT = False
MOR_AUG_RATE = 0.1
CUT_MIX_RATE = 0.9

POST_AUG_RATE = 0

USE_FOCAL_LOSS = False
USE_CLASS_BALANCED_LOSS = False
USE_LABEL_SMOOTHING = False
LS_EPSILON = 0.1

NO_EXTRA_AUG = True

USE_PRETRAINED = True
DROP_RATE = None     ###If no dp, use None instead of 0

USE_AMP = True
OPT_LEVEL = "O2"

USE_MISH = False

FINE_TUNE_EP = 10

FOLD = 0


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
from sklearn.linear_model import LinearRegression #import model
from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":

    x = np.random.random((500,1000))
    # x2 = np.random.random((5,5))
    # x = np.hstack((x,x2))
    # print(np.shape(x))
    y = np.random.randint(5,size=(500,1000))
    # print("x",x)
    print("y",y)

    enc = OneHotEncoder(sparse=False,categories="auto")
    y_onehot = enc.fit_transform(y)
    print("one hot:",y_onehot)
    print(np.shape(y_onehot))

    meta_model = LinearRegression()
    meta_model.fit(x,y_onehot)
    print("pred:",meta_model.predict(x))
    stop


    batch_size = 256
    num_workers = 12
    k = 7
    indices_len = 232560
    vr = 1/k
    print("validation rate:",vr)
    train_loaders, val_loaders = get_kfold_dataset_loader(k, vr, indices_len, batch_size, num_workers)
    if USE_FOCAL_LOSS == True:
        criterion = FocalLossWithOutOneHot(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("Fold:",FOLD)
    ensemble_root = "./stacking_model/"
    ensemble_models = []
    data_num = 0
    acc = 0 

    for file_name in os.listdir(ensemble_root):
        print(file_name)
        model = get_model(model_type=MODEL_TYPE,pretrained=False)
        if USE_APEX == True:
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
    meta_model_r=LinearRegression()
    meta_model_v=LinearRegression()
    meta_model_c=LinearRegression()
    

    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            img, target = data
            img, target = img.to(device), target.to(device,dtype=torch.long)
            pred_root, pred_vowel, pred_constant = model(img)  #(batch_num,167)


    ###stacked_pred:(data_num, 167,5), ###targets:(data_num,167)
    meta_model_r.fit(stacked_pred,targets)

    #     acc_root = 0
    #     acc_vowel = 0
    #     acc_constant = 0
    #     acc = 0
    #     val_loss_root = 0
    #     val_loss_vowel = 0
    #     val_loss_constant = 0
    #     val_loss = 0
    #     data_num  = 0
    #     with torch.no_grad():
    #         for idx, data in enumerate(val_loader):
    #             img, target = data
    #             img, target = img.to(device), target.to(device,dtype=torch.long)
    #             tmp = model(img)
    #             pred_root, pred_vowel, pred_constant = model(img)

    #             val_loss_root += criterion(pred_root, target[:,0]).item()
    #             val_loss_vowel += criterion(pred_vowel, target[:,1]).item()
    #             val_loss_constant += criterion(pred_constant, target[:,2]).item()

    #             # print(pred) 
    #             _,pred_class_root = torch.max(pred_root.data, 1)
    #             _,pred_class_vowel = torch.max(pred_vowel.data, 1)
    #             _,pred_class_constant = torch.max(pred_constant.data, 1)

    #             ###Origin metric
    #             acc_root += (pred_class_root == target[:,0]).sum().item()
    #             acc_vowel += (pred_class_vowel == target[:,1]).sum().item()
    #             acc_constant += (pred_class_constant == target[:,2]).sum().item()
    #             data_num += img.size(0)

    #     ###Origin metric
    #     acc_root /= data_num
    #     acc_vowel /= data_num
    #     acc_constant /= data_num
    #     val_loss_root /= data_num
    #     val_loss_vowel /= data_num
    #     val_loss_constant /= data_num
    #     acc = (2*acc_root + acc_vowel + acc_constant)/4
    #     val_loss = (2*val_loss_root + val_loss_vowel + val_loss_constant)/4

    #     ###Origin metric
    #     print("Val Ep{},Loss:{:.6f},rl{:.4f},vl{:.4f},cl{:.4f},Acc:{:.4f}%,ra:{:.4f}%,va:{:.4f}%,ca:{:.4f}%,lr:{}"
    #             .format(ep,val_loss,val_loss_root,val_loss_vowel,val_loss_constant,acc*100,acc_root*100,acc_vowel*100,acc_constant*100,optimizer.param_groups[0]['lr']))

            
    # ###K-Fold ensemble: Saved k best model for k dataloader
    # print("===================Best Fold:{} Saved Loss:{} Acc:{}==================".format(FOLD,min_loss,max_acc))
    # torch.save(best_model_dict, "{}_Fold{}_loss{:.4f}_acc{:.3f}".format(save_file_name,FOLD,min_loss*1e3,max_acc*1e2))
    # print("======================================================")

    # del model
    # torch.cuda.empty_cache()


    # batch_size = 128
    # num_workers = 2
    # target=[] # model predictions placeholder
    # row_id=[] # row_id place holder
    # ensemble_root = "./stacking_model/"
    # ensemble_models = []

    # data_num = 0
    # acc = 0 
    # for file_name in os.listdir(ensemble_root):
    #     print(file_name)
    #     model = get_model(model_type=MODEL_TYPE,pretrained=False)
    #     if USE_APEX == True:
    #         model = amp.initialize(model,None,opt_level="O2",keep_batchnorm_fp32=True,verbosity=0,loss_scale="dynamic")
    #     model.load_state_dict(torch.load("{}/{}".format(ensemble_root,file_name)))
    #     model.eval()
    #     ensemble_models.append(model)


    # model_num = len(ensemble_models)
    # # model_num = 1
    # print("len of models:",model_num)    
    # result = np.array([])
    # label = np.array([])

    # # datadir = featherdir = "/kaggle/input/bengali-ensemble-v2/"
    # datadir = featherdir = Dataset_root
    # par_indices = [[0], [1], [2], [3]]
    # for parquet_id in par_indices:
    #     test_images, test_img_id = prepare_image(datadir, featherdir, data_type='test', submission=True, indices=parquet_id)
    # #     test_images, test_img_id = prepare_image(datadir, featherdir, data_type='test', submission=False, indices=[0])
    #     test_dataset = TestDataset(data_len=None)
    #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)    
    #     with torch.no_grad():
    #         for idx, data in enumerate(test_loader):
    #             ###Origin
    #             img, img_id = data
    #             img = img.to(device)
                
    #             ###Average Ensemble
    #             pred_list_root = torch.Tensor([]).to(device)
    #             pred_list_vow = torch.Tensor([]).to(device)
    #             pred_list_const = torch.Tensor([]).to(device)
                            
    #             ###No TTA
    #             for model_i in range(model_num):
    #                 if model_i == size256_id:
    #                     img = img2.to(device)
    #                 else:
    #                     img = img1.to(device)
                    
    #                 pred_root, pred_vow, pred_const = ensemble_models[model_i](img) #(batch_num, label_num)
    #                 pred_list_root = torch.cat((pred_list_root,pred_root.unsqueeze(2)),dim=2)      #pred_list: (batch_num,168,model_num*tta)
    #                 pred_list_vow = torch.cat((pred_list_vow,pred_vow.unsqueeze(2)),dim=2)         #pred_list: (batch_num,11,model_num*tta)
    #                 pred_list_const = torch.cat((pred_list_const,pred_const.unsqueeze(2)),dim=2)   #pred_list: (batch_num,7,model_num*tta)                        
                
    #             pred_root = torch.mean(pred_list_root,dim=2)   #(batch,10)
    #             pred_vow = torch.mean(pred_list_vow,dim=2)   #(batch,10)
    #             pred_const = torch.mean(pred_list_const,dim=2)   #(batch,10)

    #             _,pred_class_root = torch.max(pred_root.data, 1)   #(batch_num,)        
    #             _,pred_class_vow = torch.max(pred_vow.data, 1)   #(batch_num,)        
    #             _,pred_class_const = torch.max(pred_const.data, 1)   #(batch_num,)        

    #             for i,test_id in enumerate(img_id):
    #                 row_id.append(test_id+'_consonant_diacritic')
    #                 row_id.append(test_id+'_grapheme_root')
    #                 row_id.append(test_id+'_vowel_diacritic')
    #                 target.append(pred_class_const[i].cpu().numpy())
    #                 target.append(pred_class_root[i].cpu().numpy())
    #                 target.append(pred_class_vow[i].cpu().numpy())
                    
    #     del test_images
    #     del test_img_id
        
    # df_sample = pd.DataFrame(
    #     {
    #         'row_id': row_id,
    #         'target':target
    #     },
    #     columns = ['row_id','target'] 
    # )
    # df_sample.to_csv('submission.csv',index=False)        
