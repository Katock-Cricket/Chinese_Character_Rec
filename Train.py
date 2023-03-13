import os
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Arguments import Args
from Data import MyDataset, classes_txt
from EfficientNetV2 import efficientnetv2_s

def train():
    print("Train Task")
    transform = transforms.Compose(
        [transforms.Resize((Args.img_size, Args.img_size)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         transforms.ColorJitter()])
    train_set = MyDataset(Args.data_root + 'train.txt', num_class=Args.num_class, transforms=transform)
    train_loader = DataLoader(train_set, batch_size=Args.batch_size, shuffle=True)
    device = torch.device('cuda:0')
    model = efficientnetv2_s(num_classes=Args.num_class)
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    print("load model ok")

    if has_log_file(Args.log_root):
        max_log = find_max_log(Args.log_root)
        print("continue training with " + max_log + "...")
        checkpoint = torch.load(max_log)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            param_group["lr"] = Args.lr
        loss = checkpoint['loss']
        epoch = checkpoint['epoch'] + 1
    else:
        print("train for the first time...")
        loss = 0.0
        epoch = 0

    while epoch < Args.epoch:
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
                print('epoch %5d: batch: %5d, loss: %8f, lr: %f' % (
                    epoch + 1, i + 1, running_loss / 200, optimizer.state_dict()['param_groups'][0]['lr']))
                running_loss = 0.0

        scheduler.step(loss)
        print('Save checkpoint...')
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                   Args.log_root + 'log' + str(epoch) + '.pth')
        print('Saved')
        epoch += 1

    print('Finish training')


def has_log_file(log_root):
    file_names = os.listdir(log_root)
    for file_name in file_names:
        if file_name.startswith('log'):
            return True
    return False


def find_max_log(log_root):
    files = os.listdir(log_root)
    pattern = r'log(\d+)\.pth'
    max_num = 0
    for file in files:
        match = re.match(pattern, file)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    #
    return os.path.join(log_root, f"log{max_num}.pth")


if __name__ == '__main__':
    if not os.path.exists(Args.data_root + 'train.txt'):
        classes_txt(Args.data_root + 'train', Args.data_root + 'train.txt', Args.num_class)

    train()
