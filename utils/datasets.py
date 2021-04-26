import glob
import random
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import augment
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = ((pad1, pad2), (0,0), (0,0)) if h <= w else ((0,0), (pad1, pad2), (0,0))
    img = np.pad(img, pad, 'constant', constant_values=pad_value)

    return img, (*pad[1], *pad[0])  


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = Image.open(img_path)
        img = np.array(img)
        img, _ = pad_to_square(img, 0)
        img = transforms.ToTensor()(img)
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        if len(img.shape) != 3:
            img = img[None, :, :]
            img = img.repeat(3, 0)

        h, w, _ = img.shape  
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        img, pad = pad_to_square(img, 0)
        padded_h, padded_w, _ = img.shape

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        assert os.path.exists(label_path)  
        boxes = np.loadtxt(label_path).reshape(-1, 5)
        x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
        x1 += pad[0]   
        y1 += pad[2]
        x2 += pad[0]
        y2 += pad[2]
        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        boxes[:, 3] *= w_factor / padded_w      
        boxes[:, 4] *= h_factor / padded_h

        if self.augment:
            img, boxes = augment(img, boxes)

        img = transforms.ToTensor()(img)

        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = torch.from_numpy(boxes)  

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        for boxes in targets:
            assert (boxes is not None)
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        if self.multiscale:
            imgs = torch.stack([resize(img, self.max_size) for img in imgs])   
        else:
            imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)

    def select_new_img_size(self):
        self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

    def resize_imgs(self, images):
        if self.multiscale:
            images = F.interpolate(images, size=self.img_size, mode="nearest")

        return images
