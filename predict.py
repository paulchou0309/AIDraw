from __future__ import print_function, division

import json
import time
import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, models, transforms


class_names = ['bee', 'bird', 'butterfly', 'flower', 'grass', 'leaf', 'rabbit', 'tree']

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_data(img_pos): 
    image = Image.open(img_pos).convert('RGB')
    image = data_transform(image).float()
    image.unsqueeze_(0)
    image = Variable(image)
    return image


def predict(inputs):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)

    if os.path.exists('forest.pkl'):
        model.load_state_dict(torch.load('forest.pkl'))

    model.train(False)

    outputs = model(inputs)
    _, preds = torch.topk(outputs.data, 6)
    return [class_names[x] for x in preds[0]]


def main():
    if len(sys.argv) > 1:
        img_pos = sys.argv[1]
    else:
        print('You must specify the image position.')
        return
    res = predict(load_data(img_pos))
    print(json.dumps(res))


if __name__ == '__main__':
    main()
