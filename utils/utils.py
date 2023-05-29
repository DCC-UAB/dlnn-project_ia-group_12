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
import matplotlib.pyplot as plt
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
from matplotlib.pyplot import subplots
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report







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

    total_images = total_images = 100000  # Cambiado a 5000 para equilibrar las clases (2500 benignos y 2500 malignos)

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


def train_logits (model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.unsqueeze(1))
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


def evaluate_logits(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target.unsqueeze(1)).item()
            pred = (output > 0.0).long()
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


def training_model_logits(model, device, criterion, optimizer,X_train, y_train, X_val, y_val):
    batch_size = 32  # Define the batch size
    train_dataset = TensorDataset(torch.from_numpy(X_train).float().permute(0, 3, 1, 2), torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float().permute(0, 3, 1, 2), torch.from_numpy(y_val).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 40
    best_val_loss = float('inf')
    losses = {"train": [], "val": []}
    hist = {"train": [], "val": []}

    for epoch in range(num_epochs):
        train_loss = train_logits(model, device, train_loader, optimizer, criterion)
        val_loss, val_accuracy = evaluate_logits(model, device, val_loader, criterion)

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

def evaluate_ontestset(X_test, y_test, batch_size, model, device, criterion):
    test_dataset = TensorDataset(torch.from_numpy(X_test).float().permute(0, 3, 1, 2), torch.from_numpy(y_test).long())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss, test_accuracy = evaluate(model, device, test_loader, criterion)

    print('Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_loss, test_accuracy))
    return test_loader

def evaluate_ontestset2(X_test, y_test, batch_size, model, device, criterion):
    test_dataset = TensorDataset(torch.from_numpy(X_test).float().permute(0, 3, 1, 2), torch.from_numpy(y_test).long())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss, test_accuracy, y_true, y_pred = evaluate(model, device, test_loader, criterion)

    print('Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_loss, test_accuracy))

    # Calculate recall and F1 score
    recall = np.mean(y_true == y_pred)
    f1_score = 2 * (test_accuracy * recall) / (test_accuracy + recall)

    print('Test Recall: {:.4f}, Test F1 Score: {:.4f}'.format(recall, f1_score))
    return test_loader

def evaluate_ontestset_logits(X_test, y_test, batch_size, model, device, criterion):
    test_dataset = TensorDataset(torch.from_numpy(X_test).float().permute(0, 3, 1, 2), torch.from_numpy(y_test).float())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss, test_accuracy = evaluate_logits(model, device, test_loader, criterion)

    print('Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_loss, test_accuracy))
    return test_loader


def evaluate_modelRESNET(device, criterion, model, dataloader):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()

            true_labels.extend(labels.tolist())
            pred_labels.extend(preds.tolist())

    loss = running_loss / len(dataloader.dataset)
    accuracy = running_corrects / len(dataloader.dataset)
    
    # Compute recall and F1 score
    report = classification_report(true_labels, pred_labels)
    print("Classification Report:\n", report)

    return loss, accuracy



import random

def visualizing_results(test_loader, device, model, N_IDC, P_IDC, num_images):
    list_img_names = []

    indices = list(range(len(N_IDC)))
    random.shuffle(indices)

    counter = 0
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        for j in range(inputs.size(0)):
            if counter >= num_images:
                # If the desired number of images has been printed, exit the loop
                break

            index = indices[counter]
            image_path = N_IDC[index] if preds[j].item() == 0 else P_IDC[index]
            image_name = os.path.basename(image_path)
            print("Nombre de la imagen: {}".format(image_name))
            list_img_names.append(image_name)
            
            # Loading and showing the image
            image = inputs[j].permute(1, 2, 0).cpu().numpy()

            # Normalizing the image
            image = (image - image.min()) / (image.max() - image.min())

            plt.figure()
            plt.imshow(image)
            plt.axis('off')
            plt.show()

            # Print the prediction and the correct label
            prediction = preds[j].item()
            correct_label = labels[j].item()
            print("Predicción: {}, Etiqueta correcta: {}".format(prediction, correct_label), "\n")

            counter += 1

        if counter >= num_images:
            # If the desired number of images has been printed, exit the loop
            break

    return list_img_names



def get_cancer_dataframe(base_path, patient_id, cancer_id):
    path = os.path.join(base_path, patient_id, cancer_id)
    files = os.listdir(path)
    dataframe = pd.DataFrame(files, columns=["filename"])
    path_names = [os.path.join(path, filename) for filename in dataframe.filename.values]
    dataframe = dataframe.filename.str.rsplit("_", n=4, expand=True)
    dataframe.loc[:, "target"] = np.int(cancer_id)
    dataframe.loc[:, "path"] = path_names
    dataframe = dataframe.drop([0, 1, 4], axis=1)
    dataframe = dataframe.rename({2: "x", 3: "y"}, axis=1)
    dataframe.loc[:, "x"] = dataframe.loc[:,"x"].str.replace("x", "", case=False).astype(np.int)
    dataframe.loc[:, "y"] = dataframe.loc[:,"y"].str.replace("y", "", case=False).astype(np.int)
    return dataframe

def get_patient_dataframe(base_path, patient_id):
    df_0 = get_cancer_dataframe(base_path,patient_id, "0")
    df_1 = get_cancer_dataframe(base_path, patient_id, "1")
    patient_df = pd.concat([df_0, df_1], ignore_index=True)
    return patient_df


def visualise_breast_tissue_base(base_path, patient_id, pred_df=None):
    example_df = get_patient_dataframe(base_path, patient_id)
    max_point = [example_df.y.max()-1, example_df.x.max()-1]
    grid = 255*np.ones(shape=(max_point[0] + 50, max_point[1] + 50, 3)).astype(np.uint8)
    mask = 255*np.ones(shape=(max_point[0] + 50, max_point[1] + 50, 3)).astype(np.uint8)
    if pred_df is not None:
        patient_df = pred_df[pred_df.patient_id == patient_id].copy()
    mask_proba = np.zeros(shape=(max_point[0] + 50, max_point[1] + 50, 1)).astype(np.float)
    
    broken_patches = []
    for n in range(len(example_df)):
        try:
            image = imread(example_df.path.values[n])
            target = example_df.target.values[n]
            x_coord = np.int(example_df.x.values[n])
            y_coord = np.int(example_df.y.values[n])
            x_start = x_coord - 1
            y_start = y_coord - 1
            x_end = x_start + 50
            y_end = y_start + 50

            grid[y_start:y_end, x_start:x_end] = image
            if target == 1:
                mask[y_start:y_end, x_start:x_end, 0] = 250
                mask[y_start:y_end, x_start:x_end, 1] = 0
                mask[y_start:y_end, x_start:x_end, 2] = 0
            if pred_df is not None:
                proba = patient_df[(patient_df.x==x_coord) & (patient_df.y==y_coord)].proba
                mask_proba[y_start:y_end, x_start:x_end, 0] = np.float(proba)

        except ValueError:
            broken_patches.append(example_df.path.values[n])
    
    return grid, mask, broken_patches, mask_proba


def visualise_breast_tissue(base_path, patient_id):
    grid, mask, broken_patches,_ = visualise_breast_tissue_base(base_path, patient_id)

    fig, ax = plt.subplots(1,2,figsize=(20,10))
    ax[0].imshow(grid, alpha=0.9)
    ax[1].imshow(mask, alpha=0.8)
    ax[1].imshow(grid, alpha=0.7)
    ax[0].grid(False)
    ax[1].grid(False)
    for m in range(2):
        ax[m].set_xlabel("X-coord")
        ax[m].set_ylabel("Y-coord")
    ax[0].set_title("Breast tissue slice of patient: " + patient_id)
    ax[1].set_title("Cancer tissue colored red \n of patient: " + patient_id)

    plt.show()


def visualise_breast_tissue_binary(base_path, patient_id):
        
    fig, ax = plt.subplots(1, 1)

    example_df = get_patient_dataframe(base_path, patient_id)

    ax.scatter(example_df.x.values, example_df.y.values, c=example_df.target.values, cmap="coolwarm", s=20)
    ax.set_title("Patient " + patient_id)
    ax.set_xlabel("X coord")
    ax.set_ylabel("Y coord")
    ax.set_aspect('equal')  # Set aspect ratio to 'equal' to preserve original orientation
    ax.invert_yaxis()  # Reverse the y-axis direction

    plt.show()


def initialize_data(input_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    print("Initializing Datasets and Dataloaders...")
    return data_transforms


def create_dataloaders(main_folder, data_transforms, subset_percentage=0.3, batch_size=8):
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(main_folder, x), data_transforms[x]) for x in ['train', 'val', 'test']}

    # Generate indices for the subset
    subset_indices_train = torch.randperm(len(image_datasets['train']))[:int(subset_percentage * len(image_datasets['train']))]
    subset_indices_val = torch.randperm(len(image_datasets['val']))[:int(subset_percentage * len(image_datasets['val']))]
    subset_indices_test = torch.randperm(len(image_datasets['test']))[:int(subset_percentage * len(image_datasets['test']))]

    # Create subsets
    train_data_subset = torch.utils.data.Subset(image_datasets['train'], subset_indices_train)
    val_data_subset = torch.utils.data.Subset(image_datasets['val'], subset_indices_val)
    test_data_subset = torch.utils.data.Subset(image_datasets['test'], subset_indices_test)

    # Create training and validation dataloaders
    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(train_data_subset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': torch.utils.data.DataLoader(val_data_subset, batch_size=batch_size, shuffle=True, num_workers=4),
        'test': torch.utils.data.DataLoader(test_data_subset, batch_size=batch_size, shuffle=True, num_workers=4),
    }

    return  dataloaders_dict, image_datasets


def train_modelresnet(device, model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    acc_history = {"train": [], "val": []}
    losses = {"train": [], "val": []}

    # we will keep a copy of the best weights so far according to validation accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    losses[phase].append(loss.item())

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).cpu().numpy()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            acc_history[phase].append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, acc_history, losses

def initialize_resnet(num_classes):
    # Resnet18 
    model = models.resnet18()
    
    model.fc = nn.Linear(512, num_classes)
    
    input_size = 224
        
    return model, input_size


def initialize_mobilnet(num_classes):
    # Resnet18 
    model = models.mobilenet_v2(pretrained = True)
    
    # Modify the classifier to output a single value (0 or 1)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, num_classes)
    )
    
    input_size = 224
        
    return model, input_size



def train_modelMOBILNET(device, model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    acc_history = {"train": [], "val": []}
    losses = {"train": [], "val": []}

    # we will keep a copy of the best weights so far according to validation accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    losses[phase].append(loss.item())

                    preds = torch.argmax(outputs, dim=1)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item() #running_corrects += torch.sum(preds == labels.data).cpu().numpy()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            acc_history[phase].append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, acc_history, losses