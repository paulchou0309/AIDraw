from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import sys
from PIL import Image


data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# data_dir = 'png_data'
# test_dataset = datasets.ImageFolder(
#     os.path.join(data_dir, 'test'), data_transform)
# dataloader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=1, shuffle=True)

class_names = ['bee', 'bird', 'butterfly', 'flower', 'grass', 'rabbit']

use_gpu = torch.cuda.is_available()

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_predict_inputs(image_file):
    """
    this will make a image to PyTorch inputs, as the same with training images.
    this will return a tensor, default not using CUDA.
    :param image_file:
    :return:
    """
    image = pil_loader(image_file)
    image_tensor = data_transform(image).float()
    image_tensor.unsqueeze_(0)
    return Variable(image_tensor)


def predict_single_image(image_file):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)

    model = model.cuda() if use_gpu else model
    criterion = nn.CrossEntropyLoss()

    since = time.time()

    if os.path.exists('transfer.pkl'):
        model.load_state_dict(torch.load('transfer.pkl'))
    else:
        print('Model file does not exst.')
    
    inputs = make_predict_inputs(image_file)
    inputs = inputs.cuda() if use_gpu else inputs
    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)
    return class_names[preds[0]]


def predict():
    if len(sys.argv) > 1:
        print('predict image from : {}'.format(sys.argv[1]))
        if os.path.exists(sys.argv[1]):
            image_file = sys.argv[1]
            return predict_single_image(image_file)
        else:
            print('file path does not exist.')
    else:
        print('must specific image file path.')


if __name__ == '__main__':
    print(predict())
