
import torch 
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data.dataset import Dataset
import torchvision.models as models


class IMAGENET():
    def __init__(self, arch):
        self.model = models.__dict__[arch](pretrained=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model, device_ids=[0])
    
    def predict_scores(self, image):
        image = image.clamp(0, 1)
        image_copy = image.clone()
        image_copy = torch.stack([F.normalize(image_copy[i], 
                                               [0.485, 0.456, 0.406], 
                                               [0.229, 0.224, 0.225])
                                   for i in range(image.shape[0])])
        if torch.cuda.is_available():
            image = image_copy.cuda()
        output = self.model(image_copy)
        return output.data


class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.features = self._make_layers()
        self.fc1 = nn.Linear(3200,256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256,256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256,10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def _make_layers(self):
        layers=[]
        in_channels= 3
        layers += [nn.Conv2d(in_channels, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.Conv2d(64, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(64, 128, kernel_size=3),
                   nn.BatchNorm2d(128),
                   nn.ReLU()]
        layers += [nn.Conv2d(128, 128, kernel_size=3),
                   nn.BatchNorm2d(128),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        return nn.Sequential(*layers)


    def predict(self, image):
        self.eval()
        image = torch.clamp(image,0,1).view(1, 3, 32, 32)
        # image = Variable(image, volatile=True).view(1,3, 32,32)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]
    
    def predict_batch(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict

    def predict_scores(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        return output.data

class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.features = self._make_layers()
        self.fc1 = nn.Linear(1024,200)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200,200)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(200,10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def _make_layers(self):
        layers=[]
        in_channels= 1
        layers += [nn.Conv2d(in_channels, 32, kernel_size=3),
                   nn.BatchNorm2d(32),
                   nn.ReLU()]
        layers += [nn.Conv2d(32, 32, kernel_size=3),
                   nn.BatchNorm2d(32),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(32, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.Conv2d(64, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        return nn.Sequential(*layers)


    def predict(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = image.view(1,1,28,28)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]
    
    def predict_batch(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        #image = Variable(image, volatile=True)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict

    def predict_scores(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        return output.data


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out

    def predict_scores(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = image.view(-1, 1, 28, 28)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        return output.data


class SimpleMNIST(nn.Module):
    """ Custom CNN for MNIST
        stride = 1, padding = 2
        Layer 1: Conv2d 5x5x16, BatchNorm(16), ReLU, Max Pooling 2x2
        Layer 2: Conv2d 5x5x32, BatchNorm(32), ReLU, Max Pooling 2x2
        FC 10
    """
    def __init__(self):
        super(SimpleMNIST, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def predict(self, image):
        self.eval()
        image = Variable(image.unsqueeze(0))
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]


def load_model(model, filename):
    """ Load the training model """
    model.load_state_dict(torch.load(filename))

