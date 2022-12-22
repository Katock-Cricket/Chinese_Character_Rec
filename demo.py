import torch
import torchvision.transforms as transforms
from PIL import Image

from Arguments import Args
from vgg19 import VGG19


def demo(label, image_path):
    transform = transforms.Compose([transforms.Resize((Args.img_size, Args.img_size)), transforms.ToTensor()])
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    model = VGG19()
    model.eval()
    checkpoint = torch.load(Args.log_root + Args.log_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    output = model(img)
    _, pred = torch.max(output.data, 1)
    pred = int(pred)
    print('input:', label)
    print('predict:', pred)


if __name__ == '__main__':
    demo(45, './45.jpg')
