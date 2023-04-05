import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb, rgb2yuv, yuv2rgb
import glob

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader



SIZE = 256
#workers is 4 (we can try to change it)

class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE), Image.BICUBIC),
                transforms.RandomHorizontalFlip(),  # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE), Image.BICUBIC)

        self.split = split
        self.size = SIZE
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img) #An array with shape (256,256,3) and where each input is an integer ONLY.
        img_yuv = rgb2yuv(img).astype("float32")  # Converting RGB to yuv
        img_yuv = transforms.ToTensor()(img_yuv)
        y = img_yuv[[0], ...] *2. -1.#/   # Between -1 and 1
        #uv = img_yuv[[1, 2], ...] /110.  # Between -1 and 1     #Shape (2,256,256)
        u = torch.clamp(img_yuv[[1], ...], -0.436, 0.436) #Theoretically
        u = u * 1. /0.436 # change the y at the front to u
        v = torch.clamp(img_yuv[[2], ...], -0.615, 0.615)
        v = v * 1. /0.615
        #After normalising the two above you would need to 'join' them somehow to give uv = img_yuv[[1,2], ...] or the equivalent as above
        uv = torch.cat((u,v),dim=0)
        

        

        return {'Y': y, 'uv' : uv}

    def __len__(self):
        return len(self.paths)


def make_dataloaders(batch_size=16, n_workers=4, pin_memory=True, **kwargs):  # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader
