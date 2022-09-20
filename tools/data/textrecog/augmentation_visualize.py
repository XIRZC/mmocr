import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
import numpy as np
import imgaug.augmenters as iaa


class TwoCropsTransform:

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]

class SeqCLRAug(object):

    def __call__(slef, x):
        aug_pipeline = iaa.Sequential([iaa.SomeOf((1, 5),
        [
            iaa.LinearContrast((0.5, 1.0)),
            iaa.GaussianBlur((0.5, 1.5)),
            iaa.Sharpen(alpha=(0.0, 0.5),
                            lightness=(0.0, 0.5)),
            iaa.PiecewiseAffine(scale=(0.02, 0.03), mode='edge'),
            iaa.PerspectiveTransform(scale=(0.01, 0.02))
        ],
        random_order=True)])
        return aug_pipeline(images=[x])[0]


def get_train_loader():
    traindir = os.path.join('./data/mixture/SynthText/TMPST')
    
    augmentation1 = [
        transforms.Resize([32, 32]),
        # transforms.RandomApply([
        #     SeqCLRAug(),
        # ], p=0.8),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]

    augmentation2 = [
        transforms.Resize([32, 32]),
        # transforms.RandomApply([
        #     SeqCLRAug(),
        # ], p=0.8),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]

    train_dataset = datasets.ImageFolder(
        traindir,
        TwoCropsTransform(transforms.Compose(augmentation1),
                            transforms.Compose(augmentation2))
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=100, shuffle=True,
        num_workers=0, pin_memory=True, sampler=None, drop_last=True)

    return train_dataloader

def visualize_images_by_grid(batch_imgs, rows, cols):
    
    plt.figure(figsize=(8, 8), facecolor="gray")
    plt.rcParams["axes.facecolor"] = "#0D0434"
    plt.subplots_adjust(hspace=0.4, wspace=0.6)

    gs = GridSpec(rows, cols)
    for i in range(rows):
        for j in range(cols):
            plt.subplot(gs[i, j])
            # print(f"type: {type(batch_imgs[i*rows+j])}, shape: {batch_imgs[i*rows+j*cols].shape}")
            plt.imshow(batch_imgs[i*rows+j].numpy().transpose((1, 2, 0)))
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
    plt.show() 

def main():
    train_dataloader = get_train_loader()
    for i, (batch_imgs, _) in enumerate(train_dataloader):
        batch_imgs1, batch_imgs2 = batch_imgs
        visualize_images_by_grid(batch_imgs1, 10, 10)
        visualize_images_by_grid(batch_imgs2, 10, 10)
        break

if __name__ == '__main__':
    main()