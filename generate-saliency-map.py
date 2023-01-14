import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

img = cv2.imread('RGBT-CC/val/1157_T.jpg')
(success, saliencyMap) = saliency.computeSaliency(img)

## using PicaNet

from picanet.network import Unet
from picanet.dataset import CustomDataset
import torch
from torch.utils.data import DataLoader
import torchvision
import os
from tqdm import tqdm

state_dict = torch.load('notebooks/picanet/36epo_383000step.ckpt', map_location='cpu')

model = Unet()

model.load_state_dict(state_dict)

custom_dataset = CustomDataset(root_dir='notebooks/test_thermal_images')

dataloader = DataLoader(custom_dataset, 2, shuffle=False)

model=model.eval()

for i, batch in enumerate(tqdm(dataloader)):
        img = batch
        #print(img.shape)

        with torch.no_grad():
            pred, loss = model(img)
        pred = pred[5].data
        pred.requires_grad_(False)
        for j in range(img.shape[0]):
            image = pred[j].numpy()
        
            image = np.transpose(image, (2,1,0)).reshape((224,224))
        
            plt.imshow(image)
            plt.show()
            torchvision.utils.save_image(pred[j], os.path.join('C:/Users/ishap/Documents/FourthYearData/LOP4-1/notebooks/masks_test_thermal', '{}_{}.jpg'.format(i, j)))


for img_path in glob.glob('RGBT-CC\\train\\*_RGB.jpg'):
    name = img_path.split('\\')[-1].split('_')[0] + '_GrayRGB.jpg'
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'RGBT-CC\\train\\{name}', gray)




