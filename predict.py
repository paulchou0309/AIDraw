from __future__ import print_function, division

import json
import time
import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms


class_names = ['bee', 'bird', 'butterfly', 'flower', 'grass', 'leaf', 'rabbit', 'tree']

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_data():
    data_dir = sys.argv[1] if len(sys.argv) == 2 else '.'
    pred_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'predict'), data_transform)
    dataloader = torch.utils.data.DataLoader(
        pred_dataset, batch_size=1)
    return dataloader

def predict(dataloader):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)

    if os.path.exists('forest.pkl'):
        model.load_state_dict(torch.load('forest.pkl'))

    model.train(False)

    for data in dataloader:
        inputs, _ = data
        inputs = Variable(inputs)

        outputs = model(inputs)
        _, preds = torch.topk(outputs.data, 6)
        return [class_names[x] for x in preds[0]]


if __name__ == '__main__':
    res = predict(load_data())
    print(json.dumps(res))
