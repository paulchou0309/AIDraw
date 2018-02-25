from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
# import copy

plt.ion()


data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = 'png_data'
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transform)
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)
dataset_size = len(test_dataset)

use_gpu = torch.cuda.is_available()


def test_model():
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)

    model = model.cuda() if use_gpu else model
    criterion = nn.CrossEntropyLoss()

    since = time.time()

    if os.path.exists('transfer.pkl'):
        model.load_state_dict(torch.load('transfer.pkl'))

    print('Model testing starts!')

    model.train(False)

    running_loss = 0.0
    running_corrects = 0

    for data in dataloader:
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable( inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.data[0] * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    loss = running_loss / dataset_size
    acc = running_corrects / dataset_size

    print('Loss: {:.4f} Acc: {:.4f}'.format(loss, acc))

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    test_model()
