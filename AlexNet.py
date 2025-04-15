# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
import kagglehub
import pandas as pd

# data preprocessing
# compute mean
def compute_dataset_mean(dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mean = torch.zeros(3)
    total = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # [B, C, H*W]
        mean += images.mean(2).sum(0)  # mean over pixels, sum over batch
        total += batch_samples

    return (mean / total).tolist()

# compute std
def compute_dataset_std(dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    std = torch.zeros(3)
    total = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # [B, C, H*W]
        std += images.std(2).sum(0)  # std over pixels, sum over batch
        total += batch_samples

    return (std / total).tolist()

# for training transform
def get_train_transform(mean, std):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
# for testing transform
def get_test_transform(mean, std):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Normalize(mean=mean, std=std)(
                transforms.ToTensor()(crop)) for crop in crops]))
    ])
    
# compute eigenvalue and eigenvector
def compute_pca_rgb(dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # accumulate RGB values from all pixels
    rgb_values = []

    for images, _ in loader:
        # images shape: [B, 3, H, W]
        B, C, H, W = images.shape
        images = images.permute(0, 2, 3, 1).reshape(-1, 3)  # [B*H*W, 3]
        rgb_values.append(images)

    # stack into a large [N, 3] matrix
    all_pixels = torch.cat(rgb_values, dim=0)

    # center the data (subtract the mean)
    mean = torch.mean(all_pixels, dim=0)
    centered = all_pixels - mean

    # compute covariance matrix: shape [3, 3]
    cov = torch.matmul(centered.T, centered) / (centered.shape[0] - 1)

    # compute eigenvalues and eigenvectors
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # sort in descending order
    sorted_idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[sorted_idx]
    eigvecs = eigvecs[:, sorted_idx]

    return eigvals, eigvecs

# model architecture
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # 1st conv layer
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 2nd conv layer
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 3rd conv layer
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 4th conv layer
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 5th conv layer
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)  
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.classifier(x)
        return x

# training
# data augmentation
def add_pca_noise(img: Image.Image, eigval, eigvec):
    alpha = np.random.normal(0, 0.1, 3)
    noise = eigvec @ (alpha * eigval)
    img = np.asarray(img).astype(np.float32)
    for i in range(3):
        img[..., i] += noise[i]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

# optimizer and scheduler
def get_optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

def adjust_learning_rate(optimizer, epoch, schedule=[30, 60, 80]):
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

# 10-crop inference
def predict_10crop(model, img_tensor10):
    model.eval()
    with torch.no_grad():
        img_tensor10 = img_tensor10.cuda()
        outputs = model(img_tensor10)
        probs = F.softmax(outputs, dim=1)
        return probs.mean(dim=0)
    
# model and training setup
# model = AlexNet(num_classes=1000)
# for GPU
# model = nn.DataParallel(model, device_ids=[0, 1])
# model = model.cuda()

# criterion = nn.CrossEntropyLoss()
# optimizer = get_optimizer(model)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

if __name__ == "__main__":
    model = AlexNet(num_classes=1000)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
