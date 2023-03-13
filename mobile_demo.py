import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from Arguments import Args

np.set_printoptions(threshold=np.inf)


def demo(image_path):
    transform = transforms.Compose(
        [transforms.Resize((Args.img_size, Args.img_size)), transforms.ToTensor()])
    img = Image.open(image_path)
    img = transform(img)
    img = img.unsqueeze(0)
    l1 = img.flatten().numpy().tolist()
    for i in range(len(l1)):
        print("{:.6f}".format(l1[i]) + " ", end='')
        if (i + 1) % 32 == 0:
            print()
    model = torch.load('model.pt')
    model.eval()
    with torch.no_grad():
        output = model(img)
    _, pred = torch.max(output.data, 1)
    print(_)
    print(pred)
    f = open('char_dict', 'rb')
    dic = pickle.load(f)
    for cha in dic:
        if dic[cha] == int(pred):
            print('predict: ', cha)
    f.close()


if __name__ == '__main__':
    demo('./fo.jpg')
