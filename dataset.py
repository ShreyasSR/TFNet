import os
from glob import glob
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CrowdDataset(Dataset):
    """Crowd dataset."""
    def __init__(self, root_dir,c_model,stage='train', mode='RGB', transform_c=None,transform=None, gt_downsample=1):
        self.stage = stage
        self.rgb_dir = os.path.join(root_dir, stage+'_fused')
        self.dense_dir = os.path.join("/content/drive/MyDrive/CVRS/Crowd Behavior Analysis/fused", f'density_maps_normal/{self.stage}')
        self.mode = mode
        self.transform_c = transform_c
        self.transform = transform
        self.img_list = [x.split('_')[0] for x in os.listdir(f'{self.rgb_dir}')]
        self.gt_downsample = gt_downsample
        self.classifier = c_model
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_number = self.img_list[idx]
        img_path = f'/content/drive/MyDrive/CVRS/Crowd Behavior Analysis/RGBT-CC/{self.stage}/{img_number}_RGB.jpg'
        pil_image = Image.open(img_path)
        pil_image = self.transform_c(pil_image)
        pil_image = pil_image.resize(1,3, 224, 298)
        y_pred, _ = self.classifier(pil_image)
        _, top_pred = y_pred.topk(1, 1)
        top_pred = top_pred.t()

        if top_pred.item()==1:
          categ = 'b'
          img_path = f'/content/drive/MyDrive/CVRS/Crowd Behavior Analysis/RGBT-CC/{self.stage}/{img_number}_RGB.jpg'
          image = cv2.imread(img_path)
          image = cv2.resize(image, (224,224))
        if top_pred.item()==0:
          categ = 'd'
          img_path = f'/content/drive/MyDrive/CVRS/Crowd Behavior Analysis/fused/{self.stage}_fused/{img_number}_F.jpg'
          image = cv2.imread(img_path)
        dense_path = f'{self.dense_dir}/{img_number}.npy'
        
        dmap = np.load(dense_path)
        
        if self.transform:
            image = self.transform(image)
        

        if self.gt_downsample>1:
            ds_rows=int(image.shape[0]//self.gt_downsample)
            ds_cols=int(image.shape[1]//self.gt_downsample)
            img = cv2.resize(image,(ds_cols*self.gt_downsample,ds_rows*self.gt_downsample))
            img=img.transpose((2,0,1)) 
            gt_dmap=cv2.resize(dmap,(44,44))
            gt_dmap=gt_dmap[np.newaxis,:,:]*self.gt_downsample*self.gt_downsample

            img_tensor = torch.tensor(img,dtype=torch.float)
            gt_dmap_tensor = torch.tensor(gt_dmap,dtype=torch.float)

        return img_path, img_tensor, gt_dmap_tensor, categ 