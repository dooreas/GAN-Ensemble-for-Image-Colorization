from utils import *
from loss import *
from dataset import *
 
from models import *
from infer import *

import os
import glob
import time
import numpy as np

from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
Path.ls = lambda x: list(x.iterdir())

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.version.cuda)
#My incorporations
import re
import tqdm


root = "/Users/doore/Documents/Pix2pix_JAKE/datasets/mirflickr_splitted" 

train_paths = sorted(glob.glob(os.path.join(root, "train") + "/*.jpg"))
val_paths = sorted(glob.glob(os.path.join(root, "valid") + "/*.*"))
    
model_save_as = "baseline_pix2pix_without_pretrain_150"
path_to_saved_model ="/Users/doore/Documents/Pix2pix_JAKE/"+model_save_as

progress_saved_in ="/Users/doore/Documents/Pix2pix_JAKE/Progress/"+model_save_as
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/Progress/", exist_ok=True)#create the directory where we will save the progress as a text

print(len(train_paths), len(val_paths))


if __name__ == '__main__':
    ##Making Datasets and DataLoaders
    train_dl = make_dataloaders(paths=train_paths, split='train')
    val_dl = make_dataloaders(paths=val_paths, split='val')
    



    data = next(iter(train_dl))
    Ls, abs_ = data['L'], data['ab']
    print(Ls.shape, abs_.shape)
    print(len(train_dl), len(val_dl))
    
    ##TRAINING FUNCTION

    def train_model(model, train_dl, epochs, display_every=200):
        data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intervals
        for e in range(epochs):
            loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
            i = 0                                  # log the losses of the complete network
            for data in tqdm.tqdm(train_dl):
                model.setup_input(data) 
                model.optimize()
                update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
                i += 1
                if i % display_every == 0:
                    print(f"\nEpoch {e+1}/{epochs}")
                    print(f"Iteration {i}/{len(train_dl)}")
                    log_results(loss_meter_dict) # function to print out the losses
                    visualize(model, data, str(e+1)+"_"+str(i)) # function displaying the model's outputs, but that now saves the progress images after "diplay_every" iterations

            
            with open(progress_saved_in+".txt", "a") as file:
                file.write("EPOCH "+str(e+1))
                file.write('loss_D_fake: '+str(loss_meter_dict['loss_D_fake'].avg)+", "+'loss_D_real: '+str(loss_meter_dict['loss_D_real'].avg)+", "+'loss_D: '+str(loss_meter_dict['loss_D'].avg)+"\n")
                file.write("loss_G_GAN: "+str(loss_meter_dict["loss_G_GAN"].avg)+", "+ 'loss_G_L1: '+ str(loss_meter_dict['loss_G_L1'].avg)+", "+"loss_G: "+str(loss_meter_dict['loss_G'].avg))
                file.write("\n NEXT\n")


    model = MainModel()
    train_model(model, train_dl, 200)

    #Now we save the trained model with the name specified in the model_save
    torch.save(model.state_dict(), path_to_saved_model)


   

    #To use the model then you need to use (ask ChatGPT otherwise):
    
    # Define your PyTorch model
    #model = ...
    # Define the file path where the model is saved
    #PATH = "model.pth"
    # Load the saved model state dictionary into the model
    #model.load_state_dict(torch.load(PATH))