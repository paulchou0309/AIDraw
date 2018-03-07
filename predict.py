from __future__ import print_function, division

import base64
import json
import time
import os
import sys
import getopt
import torch
import torch.nn as nn
from io import BytesIO
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


def load_data(img_input): 
    image = Image.open(img_input).convert('RGB')
    image = data_transform(image).float()
    image.unsqueeze_(0)
    image = Variable(image)
    return image


def load_model(model_pos):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)

    if os.path.exists(model_pos):
        model.load_state_dict(torch.load(model_pos, map_location=lambda storage, location: storage))
    else:
        print('Model does not exist.')

    model.train(False)
    
    return model


def predict(model, inputs):
    outputs = model(inputs)
    _, preds = torch.topk(outputs.data, 6)
    return [class_names[x] for x in preds[0]]


def main():
    model_pos = 'forest.pkl'
    try:
        opts, args = getopt.getopt(sys.argv[1:], "b:p:m:", ["base64=", "path=", "model="])
    except getopt.GetoptError:
        print("Argv needed to specify the input")
    for opt, args in opts:
        if opt in ('-p', '--path'):
            inputs = load_data(args)
        elif opt in ('-b', '--base64'):
            inputs = load_data(BytesIO(base64.b64decode(args)))
        elif opt in ('-m', '--model'):
            model_pos = args
        else:
            print(opt)
            print('Invalid argv!')   
            return
    model = load_model(model_pos)
    res = predict(model, inputs)
    print(json.dumps(res))
    

if __name__ == '__main__':
    main()
    