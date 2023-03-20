import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Data import MyDataset
from EfficientNetV2.model import efficientnetv2_s
from Utils import find_max_log, has_log_file


def evaluate(args):
    print("===Evaluate EffNetV2===")
    transform = transforms.Compose(
        [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         transforms.ColorJitter()])

    model = efficientnetv2_s(num_classes=args.num_classes)
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
