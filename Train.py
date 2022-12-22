import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Arguments import Args
from Data import MyDataset, classes_txt
from vgg19 import VGG19


def change_lr_manual(optimizer, lr):
    for params in optimizer.param_groups:
        params['lr'] = lr


def train():
    transform = transforms.Compose([transforms.Resize((Args.img_size, Args.img_size)), transforms.ToTensor()])
    train_set = MyDataset(Args.data_root + 'train.txt', num_class=Args.num_class, transforms=transform)
    train_loader = DataLoader(train_set, batch_size=Args.batch_size, shuffle=True)
    device = torch.device('cuda:0')
    model = VGG19()
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    if Args.continue_train:
        checkpoint = torch.load(Args.log_root + Args.log_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # change_lr_manual(optimizer, lr=0.00005)
        loss = checkpoint['loss']
        epoch = checkpoint['epoch'] + 1
    else:
        loss = 0.0
        epoch = 0

    while epoch < Args.epoch:
        print('lr = ', optimizer.state_dict()['param_groups'][0]['lr'])
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outs = model(inputs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                print('epoch %5d: batch: %5d, loss: %f' % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

        scheduler.step(loss)
        print('Save checkpoint...')
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                   Args.log_root + 'log' + str(epoch) + '.pth')
        epoch += 1

    print('Finish training')


if __name__ == '__main__':
    if not os.path.exists(Args.data_root + 'train.txt'):
        classes_txt(Args.data_root + 'train', Args.data_root + 'train.txt', Args.num_class)

    train()
