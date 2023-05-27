import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torchvision.transforms import ToTensor
from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
import cv2
import os
from torch.utils.data import TensorDataset, DataLoader
import matplotlib as plt
%matplotlib inline
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from glob import glob
from skimage.io import imread
from os import listdir
import time
import copy
import random



def get_data(slice=1, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  equiv to slicing with [::slice] 
    sub_dataset = torch.utils.data.Subset(
      full_dataset, indices=range(0, len(full_dataset), slice))
    
    return sub_dataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader


def make(config, device="cuda"):
    # Make the data
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = ConvNet(config.kernels, config.classes).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer


def loading_data_preparing(path):
    root_dir =  path  # Ruta del directorio de imágenes

    N_IDC = []
    P_IDC = []

    for dir_name in tqdm(os.listdir(root_dir)):
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):
            negative_dir_path = os.path.join(dir_path, '0')
            positive_dir_path = os.path.join(dir_path, '1')
            if os.path.isdir(negative_dir_path) and os.path.isdir(positive_dir_path):
                negative_image_paths = [
                    os.path.join(negative_dir_path, image_name)
                    for image_name in os.listdir(negative_dir_path)
                    if image_name.endswith('.png')
                ]
                positive_image_paths = [
                    os.path.join(positive_dir_path, image_name)
                    for image_name in os.listdir(positive_dir_path)
                    if image_name.endswith('.png')
                ]
                N_IDC.extend(negative_image_paths)
                P_IDC.extend(positive_image_paths)

    total_images = 50000  # Cambiado a 5000 para equilibrar las clases (2500 benignos y 2500 malignos)

    n_img_arr = np.zeros(shape=(total_images, 50, 50, 3), dtype=np.float32)
    p_img_arr = np.zeros(shape=(total_images, 50, 50, 3), dtype=np.float32)
    label_n = np.zeros(total_images)
    label_p = np.ones(total_images)

    for i, img in tqdm(enumerate(N_IDC[:total_images])):
        n_img = cv2.imread(img, cv2.IMREAD_COLOR)
        n_img_size = cv2.resize(n_img, (50, 50), interpolation=cv2.INTER_LINEAR)
        n_img_arr[i] = n_img_size

    for i, img in tqdm(enumerate(P_IDC[:total_images])):
        p_img = cv2.imread(img, cv2.IMREAD_COLOR)
        p_img_size = cv2.resize(p_img, (50, 50), interpolation=cv2.INTER_LINEAR)
        p_img_arr[i] = p_img_size

    X = np.concatenate((p_img_arr, n_img_arr), axis=0)
    y = np.concatenate((label_p, label_n), axis=0)
    X, y = shuffle(X, y, random_state=0)

    # probar --> y = to_categorical(y)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    return N_IDC, P_IDC, total_images, n_img_arr, p_img_arr, label_n, label_p, X, y, X_train, X_test, y_train, y_test, X_val, y_val


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    return train_loss

def evaluate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader)
    accuracy = 100.0 * correct / len(val_loader.dataset)
    return val_loss, accuracy

def training_model(model, device, criterion, optimizer,X_train, y_train, X_val, y_val):
    batch_size = 32  # Define the batch size
    train_dataset = TensorDataset(torch.from_numpy(X_train).float().permute(0, 3, 1, 2), torch.from_numpy(y_train).long())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float().permute(0, 3, 1, 2), torch.from_numpy(y_val).long())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 40
    best_val_loss = float('inf')
    losses = {"train": [], "val": []}
    hist = {"train": [], "val": []}

    for epoch in range(num_epochs):
        train_loss = train(model, device, train_loader, optimizer, criterion)
        val_loss, val_accuracy = evaluate(model, device, val_loader, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        losses["train"].append(train_loss)
        losses["val"].append(val_loss)
        hist["train"].append(val_accuracy)
        hist["val"].append(val_accuracy)

        print('Epoch: {:02d}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.2f}%'.format(
            epoch, train_loss, val_loss, val_accuracy))

    # Load the best model
    model.load_state_dict(torch.load('best_model.pt'))

    return losses, hist

def plotting_loses (losses, hist):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(losses["train"], label="training loss")
    ax1.plot(losses["val"], label="validation loss")
    ax1.legend()

    ax2.plot(hist["train"], label="training accuracy")
    ax2.plot(hist["val"], label="validation accuracy")
    ax2.legend()

    plt.show()

