import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Data import MyDataset
from model import VGG19
from Utils import find_max_log, has_log_file


def evaluate(args):
    print("===Evaluate VGG19===")
    transform = transforms.Compose(
        [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])

    model = VGG19()
    model.eval()
    if has_log_file(args.log_root):
        file = find_max_log(args.log_root)
        print("Using log file: ", file)
        checkpoint = torch.load(file)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: No log file")

    model.to(torch.device('cuda:0'))
    test_loader = DataLoader(MyDataset(args.data_root + 'test.txt', num_class=args.num_classes, transforms=transform),
                             batch_size=args.batch_size, shuffle=False)
    total = 0.0
    correct = 0.0
    print("Evaluating...")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].cuda(), data[1].cuda()
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
    acc = correct / total * 100
    print('Accuracy'': ', acc, '%')
