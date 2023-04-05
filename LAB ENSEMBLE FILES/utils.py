import time

import numpy as np
from skimage.color import rgb2lab, lab2rgb

from PIL import Image

import torch
import os

#This was changed for validation> CHANGE BACK FOR NEW TRAINING
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/LAB_IMGs_per_cat",exist_ok=True)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}


def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


#Creates a per category progress of image generation:

#create validation and training folders: 
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/LAB_IMGs_per_cat/Train_Photos/", exist_ok= True)
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/LAB_IMGs_per_cat/Validation_Photos/", exist_ok= True)

#Person Category 
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/LAB_IMGs_per_cat/Train_Photos/people", exist_ok=True)
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/LAB_IMGs_per_cat/Validation_Photos/people", exist_ok=True)

#Animal category
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/LAB_IMGs_per_cat/Train_Photos/animal", exist_ok=True)
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/LAB_IMGs_per_cat/Validation_Photos/animal", exist_ok=True)

#Scenery Category
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/LAB_IMGs_per_cat/Train_Photos/scenery", exist_ok=True)
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/LAB_IMGs_per_cat/Validation_Photos/scenery", exist_ok=True)

#Others category
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/LAB_IMGs_per_cat/Train_Photos/others", exist_ok=True)
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/LAB_IMGs_per_cat/Validation_Photos/others", exist_ok=True)

#nature categroy
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/LAB_IMGs_per_cat/Train_Photos/nature", exist_ok=True)
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/LAB_IMGs_per_cat/Validation_Photos/nature", exist_ok=True)

#cities categroy
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/LAB_IMGs_per_cat/Train_Photos/city", exist_ok=True)
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/LAB_IMGs_per_cat/Validation_Photos/city", exist_ok=True)

def visualize(model, data, name_saved):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)

    #I implemented the following until the next # 
    real = np.concatenate([real_imgs[i] for i in range(3)], axis = 0)#change in this and the above 1 to 3 for training
    fake = np.concatenate([fake_imgs[i] for i in range(3)], axis = 0)
    
    progress_image = np.concatenate([real,fake], axis = 1)*255
    progress_image = progress_image.astype(np.uint8)
    
    Image.fromarray(progress_image).save("/Users/doore/Documents/Pix2pix_JAKE/LAB_IMGs_per_cat/"+name_saved+".jpg")#CHANGE THIS FOR THE MLP CLUSTER

def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = f"{loss_name}: {loss_meter.avg:.5f}"