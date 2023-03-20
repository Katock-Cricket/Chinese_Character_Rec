import pickle

import torch
import torchvision.transforms as transforms
from PIL import Image

from model import VGG19

from Utils import find_max_log, has_log_file


def demo(args):
    print('==Demo VGG19===')
    print('Input Image: ', args.demo_img)
    transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
    img = Image.open(args.demo_img)
    img = transform(img)
    img = img.unsqueeze(0)
    model = VGG19()
    model.eval()
    if has_log_file(args.log_root):
        file = find_max_log(args.log_root)
        print("Using log file: ", file)
        checkpoint = torch.load(file)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: No log file")

    with torch.no_grad():
        output = model(img)
    _, pred = torch.max(output.data, 1)
    f = open('../char_dict', 'rb')
    dic = pickle.load(f)
    for cha in dic:
        if dic[cha] == int(pred):
            print('predict: ', cha)
    f.close()
