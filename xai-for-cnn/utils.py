############################################################
##### Imports
############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

from PIL import Image
import cv2 as cv

from torch.nn import functional as F
import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim

import sys  
sys.path.append('../data_and_models/')
from model_net import Net

############################################################
##### Utility Fuctions
############################################################

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output.log(), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 
                                                                           100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output.log(), target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 
                                                                                 100. * correct / len(test_loader.dataset)))
def get_trained_model(nb_of_epochs=5, seed=2):
    torch.manual_seed(seed)
    batch_size = 128
    device = torch.device('cpu')
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist_data', train=True, download=True,
                                                      transform=transforms.Compose([transforms.ToTensor()])),
                                       batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist_data', train=False,
                                                      transform=transforms.Compose([transforms.ToTensor()])),
                                       batch_size=batch_size, shuffle=True)
    
    # instantiate the model and call the train and test functions 
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    # start the training and testing process
    for epoch in range(nb_of_epochs):
        train(model, device, train_loader, optimizer, epoch + 1)
        test(model, device, test_loader)
    return model, test_loader


def transform_img(img, mean, std, tensor_flag=True, img_size=(224, 224)):
    transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(img_size), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)])
    arr_img = np.array(img)
    # apply the transforms
    trans_img = transform(arr_img)
    # unsqueeze to add a batch dimension
    trans_img = trans_img.unsqueeze(0)
    if tensor_flag is False:
        # returns np.array with original axes
        trans_img = np.array(trans_img)
        trans_img = trans_img.swapaxes(-1,1).swapaxes(1, 2)

    return trans_img


def normalize_and_adjust_axes(image, mean, std):
    if image.max() > 1:
        image /= 255
    image = (image - mean) / std
    # in addition, roll the axes so that they suit pytorch
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()


def read_img(path_to_img):
    img = cv.imread(path_to_img) # Insert the path to image.
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def calculate_localization_map(gcmodel, img, out, c):

    # Step 1 - Gradient output y wrt. to activation map
    # get the gradient of the output with respect to the parameters of the model
    out[:,c].backward(retain_graph=True)
    # pull the gradients out of the model
    gradients = gcmodel.get_gradient()

    # Step 2 - Global average pooling
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]) #to be computed by students

    # Step 3 - Weighted combination of influence and feature maps
    # get the activations of the last convolutional layer
    activations = gcmodel.get_activations(img).detach()
    # weight the channels by corresponding gradients
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]
    # average the channels of the activations
    localization_map = torch.sum(activations, dim=1).squeeze()
    # convert the map to be a numpy array
    localization_map = localization_map.numpy()
    # relu on top of the localization map
    localization_map = np.maximum(localization_map, 0) #to be computed by students

    return localization_map


def convert_to_heatmap(localization_map, img):
    # normalize the localization_map
    localization_map /= np.max(localization_map)
    # resize to image size
    heatmap = cv.resize(localization_map, (img.shape[1], img.shape[0]))
    # normalize to [0, 255] range and convert to unsigned int
    heatmap = np.uint8(255 * heatmap)
    return heatmap


def read_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 

            
def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])    

    return transf


def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf   


def batch_predict(images, model):
    # Set the model in evaluation mode
    model.eval()
    # Prepare a batch of preprocessed images
    preprocess_transform = get_preprocess_transform()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    # Move the model and batch to the selected device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    # Make predictions using the model
    logits = model(batch)
    # Convert logits to class probabilities using softmax
    probs = F.softmax(logits, dim=1)
    # Return the predicted probabilities as a NumPy array
    return probs.detach().cpu().numpy()
    
