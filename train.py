
from __future__ import print_function 
from __future__ import division
from datetime import date
today = str(date.today())

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy

from dataset import *   
from dilated_mcnn import *
from metrics import *


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


#loading classifier
model_name = "squeezenet"
num_classes = 2
feature_extract = False
pre_trained=True
model_ft = models.squeezenet1_0(pretrained=pre_trained)
model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
model_ft.num_classes = num_classes
input_size = 224
model_ft.load_state_dict(torch.load('/content/drive/MyDrive/Crowd Behavior Analysis/classifier_checkpoints/squeezenet96.pth'))

#loading data
data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])  

img_root="/content/drive/MyDrive/CVRS/Crowd Behavior Analysis/fused"
stage='train'
mode='RGB'
transform=None
gt_downsample = 4

train_dataset = CrowdDataset(img_root,stage, mode, data_transforms, transform, gt_downsample)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataset = CrowdDataset(img_root,'test', 'RGB', data_transforms, transform, gt_downsample)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
val_dataset = CrowdDataset(img_root,'val', 'RGB',data_transforms, transform, gt_downsample)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

model_mcnn = MCNN()

torch.backends.cudnn.enabled=False
device=torch.device("cuda") 

#Without loading weights:
mcnn = MCNN().to(device)


criterion=nn.MSELoss(size_average=False).to(device)
optimizer = torch.optim.SGD(mcnn.parameters(), lr=1e-6,
                            momentum=0.95)

#training phase

last_epoch = 0
min_mae = 10000
min_mse = 10000
min_epoch = 0

for epoch in range(last_epoch+1,last_epoch+100):
    mcnn.train()
    epoch_loss=0
    for i,(path,img,gt_dmap,categ) in enumerate(train_dataloader):
        img=img.to(device)
        gt_dmap=gt_dmap.to(device)
        # forward propagation
        et_dmap=mcnn(img)
        # calculate loss
        loss=criterion(et_dmap,gt_dmap)
        epoch_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mcnn.eval()
    mae = 0
    mse = 0
    for i,(path,img,gt_dmap,categ) in enumerate(val_dataloader):
        img=img.to(device)
        
        gt_dmap=gt_dmap.to(device)
        # forward propagation
        et_dmap=mcnn(img)
        mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
        mse += ((et_dmap.data.sum()-gt_dmap.data.sum()).item())**2
        # del img,gt_dmap,et_dmap
    if mae/len(val_dataloader)<min_mae:
        min_mae=mae/len(val_dataloader)
        min_mae_epoch=epoch
    if mse/len(val_dataloader)<min_mse:
        min_mse=mse/len(val_dataloader)
        min_mse_epoch=epoch
        torch.save(mcnn.state_dict(),'/content/drive/MyDrive/CVRS/Crowd Behavior Analysis/checkpoints_sqnet-dilatedmcnn/epoch_'+str(epoch)+".param")
    
    print("epoch:"+str(epoch)+" mae:"+str(mae/len(val_dataloader))+" min_mae:"+str(min_mae)+" min_mae_epoch:"+str(min_mae_epoch))
    print("epoch:"+str(epoch)+" mse:"+str(mse/len(val_dataloader))+" min_mse:"+str(min_mse)+" min_mse_epoch:"+str(min_mse_epoch))


# testing phase
mcnn.load_state_dict(torch.load('/content/drive/MyDrive/CVRS/Crowd Behavior Analysis/checkpoints_sqnet-dilatedmcnn/epoch_'+str(min_mse_epoch)+'.param'))
game = [0, 0, 0, 0]

mcnn.eval()
mae = 0
mse = 0
for i,(path,img,gt_dmap,categ) in enumerate(test_dataloader):
    img=img.to(device)
    gt_dmap=gt_dmap.to(device)
    et_dmap=mcnn(img)
    for L in range(4):
        abs_error, square_error = eval_game(et_dmap, gt_dmap, L)
        game[L] += abs_error
    mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
    mse += ((et_dmap.data.sum()-gt_dmap.data.sum()).item())**2
    del img,gt_dmap,et_dmap


print("epoch:"+str(min_mse_epoch)+" mae:"+str(mae/len(test_dataloader)))
print("epoch:"+str(min_mse_epoch)+" mse:"+str(mse/len(test_dataloader)))

game=[x/len(test_dataloader) for x in game]

print(game)











