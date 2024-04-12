############################################################
##### Imports
############################################################

import numpy as np
import cv2 as cv
from torchvision import transforms

############################################################
##### Utility Fuctions
############################################################

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


def read_img(path_to_img):
    img = cv.imread(path_to_img) # Insert the path to image.
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img
