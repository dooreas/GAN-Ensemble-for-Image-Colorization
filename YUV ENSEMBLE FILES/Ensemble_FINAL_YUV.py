from utils import *
from loss import *
from dataset import *
from models import *

import os
import glob
import time
import numpy as np

from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2yuv, yuv2rgb
Path.ls = lambda x: list(x.iterdir())

import torch
from torch import nn, optim
from torchvision import transforms, models
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.version.cuda)

#My incorporations
import re
import tqdm

import torchvision.models as model_resnet
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from  torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics import PeakSignalNoiseRatio as PSNR
from pathlib import Path
from torchmetrics.image.inception import InceptionScore 

#The following are the directories where we will save the produced images compared with the real ones and alone
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/Ensemble_Test_YUV", exist_ok=True)
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/Ensemble_Test_YUV/Fakes", exist_ok=True)
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/Ensemble_Test_YUV/Fake_Real_Comparison", exist_ok=True)



#To meassure the inception score
inception = InceptionScore()

SIZE =256

input_size =224 #for resnet34 we resize them to this shape for our comodity

#same as in val for the ResNet34 model
transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

#same transforms as in val for the Pix2pix models
transform_2 =transforms.Compose([transforms.Resize((SIZE, SIZE), Image.BICUBIC), transforms.ToTensor()])


if __name__ == '__main__':

    #num_classes
    num_classes = 6

    #first we load the resnet34 model
    resnet = models.resnet34()
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    resnet.load_state_dict(torch.load("Resnet34.pth"))

    #set it into evaluation mode
    resnet.eval()

    #I take the same images in color for testing
    root = "/Users/doore/Documents/Pix2pix_JAKE/datasets/mirflickr_splitted_categ" 
    test_paths_cat = sorted(glob.glob(os.path.join(root,"test")+"/*.jpg"))
    test_dls = make_dataloaders(paths=test_paths_cat, split='val')





    #now we initiate all the colorization models:
    categ = [ "animal",  "city", "nature", "others", "people", "scenery"] #the indices in this list have been checked to coincide with what our resnet model predicts for instances oft he classes
    model_dict = {}
    for  category in categ:
        model_save_as = "_YUV_baseline_pix2pix_early_progress"
        path_to_saved_model ="/Users/doore/Documents/Pix2pix_JAKE/" + category + model_save_as
        model = MainModel()
        model.load_state_dict(torch.load(path_to_saved_model))
        model_dict[category] = model



    #The following list will take the L1 losses, the ssim and the PSNR of the produced images:
    G_L1 = []
    ssim_score = []
    fake_images = torch.empty(1,3,256,256) #for the inception score

    #we go through each path, open it as a PIL.Image instance, and then we use Resnet34 on it.
    i=0
    for data in test_dls: #IMPORTANT: BS=1 required.

        #open images and apply the same transformations as for the validation set
        image = Image.open(test_paths_cat[i]).convert("L")
        image_1 = np.array(image)
        image_1 = np.tile(image_1,(3,1,1))
        image_1 = image_1.T
        image_1 = Image.fromarray(np.uint8(image_1))
        image_2 = transform(image_1)
        image_2 = torch.unsqueeze(image_2, dim=0)

        #use the model without gradients 
        with torch.no_grad():
            outputs = resnet(image_2)
            _, preds = torch.max(outputs, 1)

        #the model for the colorization of our image taking into account the category is now taken.
        mod = MainModel()
        mod = (model_dict[categ[int(preds)]]).to(device)
        mod.eval()

        # We  take the original image an input it to our category model
         #CHANGE BATCH_SIZE TO 1
        mod.net_G.eval()
        with torch.no_grad():
            mod.setup_input(data)
            mod.forward()
        mod.net_G.train()
        fake_color = mod.fake_color.detach()
        real_color = mod.uv
        L = mod.Y
        fake_imgs = yuv_to_rgb(L, fake_color)
        real_imgs = yuv_to_rgb(L, real_color)

        #I implemented the following until the next # 
        real = np.concatenate([real_imgs[i] for i in range(1)], axis = 0)#change in this and the above 1 to 3 for training
        fake = np.concatenate([fake_imgs[i] for i in range(1)], axis = 0)

        #The fake image produced 
        fake = np.concatenate([fake_imgs[i] for i in range(1)], axis = 0)*255
        fake = fake.astype(np.uint8)
        Image.fromarray(fake).save("/Users/doore/Documents/Pix2pix_JAKE/Ensemble_Test_YUV/Fakes/"+categ[int(preds)]+"_"+str(i)+".jpg")

        #progress image provide a comparison between real and fake images
        progress_image = np.concatenate([real*255,fake], axis = 1)
        progress_image = progress_image.astype(np.uint8)
        Image.fromarray(progress_image).save("/Users/doore/Documents/Pix2pix_JAKE/Ensemble_Test_YUV/Fake_Real_Comparison/"+categ[int(preds)]+"_"+str(i)+".jpg")

        #We calculate the L1 Loss of our produced images:
        G_L1.append( float((mod.L1criterion(fake_color, real_color) * mod.lambda_L1).cpu()))

        #We calculate the SSIM error:
        img1 = torch.unsqueeze(torch.tensor(fake.T), dim=0).float()
        im = np.array(Image.open(test_paths_cat[i]).resize((256, 256)))
        img2 = torch.unsqueeze(torch.tensor(im.T), dim=0).float()
        ssim_score.append(float(ssim(img1, img2)))

        #for calculating the inceptiomn score afterwards
        fake_images = torch.cat([fake_images, torch.tensor(img1)], dim =0)
        
        #we free a little bit of our GPU's space
        mod.to("cpu")
        i+=1

    
    print(" The Mean L1 error:",np.mean(G_L1))
    print(" The Mean SSIM error:",np.mean(ssim_score))

    inception.update(fake_images)
    incept = inception.compute()
    print(" The Mean Inception error:", incept)





# we initially tried aftyer initializing the model
# # We convert the image to have one channel and use the same transforms as for the validation in Pix2pix
#         image = image.convert("L")
#         data = transform_2(image)/50. -1.
#         data = torch.unsqueeze(data, dim=0)
#         data = data.to(device)
                
#         #The colorization happens and we output the result of it. 
#         with torch.no_grad():
#             ab = mod.net_G(data).detach()
        
#         data = data
#         ab = ab

#         data = (data + 1.) * 50.
#         ab = ab * 110.
#         Lab = torch.cat([data, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
#         rgb_imgs = []
#         for img in Lab:
#             img_rgb = lab2rgb(img)
#             rgb_imgs.append(img_rgb)
#         rgb_im = np.squeeze(np.stack(rgb_imgs, axis=0), axis=0)
#         print(rgb_im.shape)