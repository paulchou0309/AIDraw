from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import sys


class_names = ['bee', 'bird', 'butterfly', 'flower', 'grass', 'leaf', 'rabbit', 'tree']

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_data():
    data_dir = sys.argv[1] if len(sys.argv) == 2 else '.'
    # data_dir = '.'
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
        _, preds = torch.topk(outputs.data, 4)
        return [class_names[x] for x in preds[0]]


if __name__ == '__main__':
    since = time.time()
    print(predict(load_data()))
    time_elapsed = time.time() - since
    print('Finished in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
