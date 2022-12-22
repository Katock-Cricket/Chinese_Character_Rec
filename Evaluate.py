import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Arguments import Args
from Data import MyDataset, classes_txt
from vgg19 import VGG19


def evaluate():
    transform = transforms.Compose([transforms.Resize((Args.img_size, Args.img_size)), transforms.ToTensor()])
    model = VGG19()
    model.to(torch.device('cuda:0'))
    checkpoint = torch.load(Args.log_root + Args.log_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_loader = DataLoader(MyDataset(Args.data_root + 'test.txt', num_class=Args.num_class, transforms=transform),
                             batch_size=Args.batch_size, shuffle=False)
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].cuda(), data[1].cuda()
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
    acc = correct / total * 100
    print('Accuracy'': ', acc, '%')


if __name__ == '__main__':
    if not os.path.exists(Args.data_root + 'test.txt'):
        classes_txt(Args.data_root + 'test', Args.data_root + 'test.txt', Args.num_class)

    evaluate()
