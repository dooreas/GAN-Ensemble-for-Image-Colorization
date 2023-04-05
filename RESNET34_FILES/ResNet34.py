from __future__ import print_function
from __future__ import division

import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from IPython import display
display.set_matplotlib_formats('svg')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import f1_score
import tqdm
import matplotlib.pyplot as plt

from PIL import Image

print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
import sys
sys.path.append('/Users/doore/Documents/Pix2pix_JAKE/datasets/mirflickr_splitted_categ_for_resnet/')
# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "/Users/doore/Documents/Pix2pix_JAKE/datasets/mirflickr_splitted_categ_for_resnet/"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 6

# Batch size for training (change depending on how much memory you have)
batch_size = 16

# Number of epochs to train for
num_epochs = 100

# Flag for feature extracting. When False, we finetune the whole model,
# when True we only update the reshaped layer params
feature_extract = False

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):

    since = time.time()

    val_f1_history = [] #Empty list to keep a running history of our F1 Score.
    train_f1_history =[]



    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0 # Initial value to compare future F1 Scores to and keep a running best score.

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        epoch_preds = [] #Empty list defined to keep a list of the predictions made for use in the F1 Score function later.
        epoch_labels = [] #Empty list defined to keep a list of the labels for use in the F1 Score function later.

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm.tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)


                    _, preds = torch.max(outputs, 1)
                    epoch_preds = epoch_preds + preds.tolist() #Required a list 
                    
                    # backward + optimize only if in training phase
                    if phase == 'train': #Let's learn from the preds
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0) #input.size(0) = brach size
                running_corrects += torch.sum(preds == labels.data) #torch.sum(preds == labels.data) is the number of correct predictions (tensor(number))
                epoch_labels = epoch_labels + labels.data.tolist() #
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(len(dataloaders['train']))
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            f1_score_epoch = f1_score(epoch_labels, epoch_preds, average='macro') #Finally computing the F1 Score for each epoch.

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)) #Keep tabs on which epoch we are on.
            print(f1_score_epoch)

            # Determine the 'best' epoch based on the best F1 Score.
            if phase == 'valid' and f1_score_epoch > best_f1: 
                best_f1 = f1_score_epoch
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'valid':
                val_f1_history.append(f1_score_epoch)

            if phase =="train":
                train_f1_history.append(f1_score_epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best F1 score: {:4f}'.format(best_f1))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_f1_history, train_f1_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

#(fc): Linear(in_features=512, out_features=8, bias=True) #Define fully connected layers.
#model.fc = nn.Linear(512, num_classes)


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet": #We will only be using ResNet in this notebook.
        """ Resnet18
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)

# Print the model we just instantiated
#print(model_ft)
#Print the model we just instantiated
print(model_ft)
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}

# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid']}


#The next commanded out lines where to discober the label-class correspondance of our datasets instance
# print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
# dist=[]
# im =[]
# for i, label in image_datasets["train"]:
#     if len(dist)==0: dist.append(label), im.append(i)
#     elif len(dist)>0 and label not in dist: dist.append(label), im.append(i)

# for j in range(len(dist)):
#     print(dist[j])
#     print(im[j])
#     torchvision.transforms.functional.to_pil_image(im[j]).show()
#     # Image.fromarray(np.uint8(((im[j].numpy()).T)*255).clip(0,255)).show()

#to here the code was written to know the class to label correspondence

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.00075, momentum=0.9) #These are the parameters we will be changing in this notebook. Set to the 'BEST' model parameters currently.

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_f1_2, _val_hist, train_hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

torch.save(model_f1_2.state_dict(), "Resnet34.pth")

fig, ax = plt.subplots()
plt.plot(range(num_epochs), train_hist, "r", label="Train")
plt.plot(range(num_epochs), _val_hist, "b", label="Validation")
plt.ylim(0,1)
plt.xlabel("Epoch")
plt.ylabel("F1-score")
plt.legend()
plt.title("Progress of Train and Validation F1-score")
plt.savefig("Train_Val_F1-score_Resnet34.jpg")


# joblib.dump(model_f1_2,"model_f1_2.joblib") #Saving the model with joblib for future reference.
print('Done')