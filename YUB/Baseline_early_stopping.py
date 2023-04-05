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
from skimage.color import rgb2lab, lab2rgb, yuv2rgb, rgb2yuv
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
val_paths = sorted(glob.glob(os.path.join(root, "valid") + "/*.jpg"))
    
model_save_as = "baseline_YUB_pix2pix_without_pretrain_early_progress"
path_to_saved_model ="/Users/doore/Documents/Pix2pix_JAKE/"+model_save_as

progress_saved_in ="/Users/doore/Documents/Pix2pix_JAKE/Progress/"+model_save_as
os.makedirs("/Users/doore/Documents/Pix2pix_JAKE/Progress/", exist_ok=True)#create the directory where we will save the progress as a text

print(len(train_paths), len(val_paths))


if __name__ == '__main__':
    ##Making Datasets and DataLoaders
    train_dl = make_dataloaders(paths=train_paths, split='train')
    val_dl = make_dataloaders(paths=val_paths, split='val')

    ##TRAINING FUNCTION
    def train_model(model, train_dl, epochs, display_every=200):
        early=np.empty((1,3))
        stop=np.empty((1,2))
        discr_val=[]
        for e in range(epochs):
            loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
            i = 0   

            # log the losses of the complete network
            for data in tqdm.tqdm(train_dl):
                model.setup_input(data) 
                model.optimize()
                update_losses(model, loss_meter_dict, count=data['Y'].size(0)) # function updating the log objects
                i += 1
                if i % display_every == 0:
                    print(f"\nEpoch {e+1}/{epochs}")
                    print(f"Iteration {i}/{len(train_dl)}")
                    log_results(loss_meter_dict) # function to print out the losses
                    visualize(model, data, "Train_Photos/"+str(e+1)+"_"+str(i)) # function displaying the model's outputs, but that now saves the progress images after "diplay_every" iterations
            
            #Validation losses are calculated now
            print("VALIDATION\n")
            u=0
            for data in tqdm.tqdm(val_dl):
                model.eval()
                model.setup_input(data)
                with torch.no_grad():
                    model.forward()

                    #first obtain the fake and real images
                    fake_image = torch.cat([model.Y, model.fake_color], dim=1) 
                    real_image = torch.cat([model.Y, model.uv], dim=1)
                

                    #we calculate the Discriminator loss
                    fake_preds = model.net_D(fake_image.detach())
                    loss_D_fake = model.GANcriterion(fake_preds, False)
                    real_preds = model.net_D(real_image)
                    fake_image.to("cpu")
                    real_image.to("cpu")
                    loss_D_real = model.GANcriterion(real_preds, True)
                    loss_D = (loss_D_fake + loss_D_real) * 0.5
                

                    #we do the same but with the Generator
                    loss_G_GAN = model.GANcriterion(fake_preds, True)
                    loss_G_L1 = model.L1criterion(model.fake_color, model.uv) * model.lambda_L1
                    loss_G = loss_G_GAN + loss_G_L1
                    #we save the data
                    early = np.append(early,[[loss_G_L1.to("cpu").numpy(),loss_G_GAN.to("cpu").numpy(),loss_G.to("cpu").numpy()]],axis=0)
                    discr_val.append(loss_D.to("cpu").numpy())


                    u+=1
                    #now we display the validation images for this epoch
                    if u % 40 == 0:
                        visualize(model, data, "Validation_Photos/"+str(e+1)+"_"+str(u))
            
            
            with open(progress_saved_in+".txt", "a") as file:
                file.write("EPOCH "+str(e+1)+"\n")
                file.write("Training_data:")
                file.write('loss_D_fake: '+str(loss_meter_dict['loss_D_fake'].avg)+", "+'loss_D_real: '+str(loss_meter_dict['loss_D_real'].avg)+", "+'loss_D: '+str(loss_meter_dict['loss_D'].avg)+"\n")
                file.write("loss_G_GAN: "+str(loss_meter_dict["loss_G_GAN"].avg)+", "+ 'loss_G_L1: '+ str(loss_meter_dict['loss_G_L1'].avg)+", "+"loss_G: "+str(loss_meter_dict['loss_G'].avg))
                file.write("\n Validation_data:")
                file.write("Loss_D: "+str(np.mean(discr_val[-u:-1]))+", Loss_G: "+str(np.mean(early[-u:,2])))
                file.write(", Loss_G_Gan: "+ str(np.mean(early[-u:,1]))+", Loss_G_L1: "+str(np.mean(early[-u:,0])))
                file.write("\n NEXT\n")
            

            print("Current mean G_L1: ", np.mean(early[-u:,0]))
            print("Current mean G_GAN: ", np.mean(early[-u:,1]))

            #Early Stopping with Progress accounting Loss_G
            
            k=10 #k defines the amount of epochs over which you will account progress
            alpha = 0.75 #alpha defines the bound over which we  stop when it is over it
            stop = np.append(stop,[[loss_meter_dict['loss_G'].avg, np.mean(early[-u:,2]) ]], axis =0)
            gener_error = 100 *((stop[-1,1]/np.min(stop[:,1]))-1)
            progress_vel = 1000*((np.mean(stop[-k:,0])/np.min(stop[-k:,0]))-1)

            if e>k:
                 if gener_error/progress_vel>alpha: break


    model = MainModel()
    train_model(model, train_dl, 150)

    #Now we save the trained model with the name specified in the model_save
    torch.save(model.state_dict(), path_to_saved_model)
    #torch.save(model.net_G.state_dict(), path_to_saved_model+"_Generator")
    #torch.save(model.net_D.state_dict(), path_to_saved_model+"_Discriminator")

   

    #To use the model then you need to use (ask ChatGPT otherwise):
    
    # Define your PyTorch model
    #model = ...
    # Define the file path where the model is saved
    #PATH = "model.pth"
    # Load the saved model state dictionary into the model
    #model.load_state_dict(torch.load(PATH))
