from PIL import Image
from torch.utils.data import Dataset
import os


class MyDataset(Dataset):
    def __init__(self, txt_path, num_class, transforms=None):
        super(MyDataset, self).__init__()
        images = []
        labels = []
        with open(txt_path, 'r') as f:
            for line in f:
                if int(line.split('\\')[1]) >= num_class:  # just get images of the first #num_class
                    break
                line = line.strip('\n')
                images.append(line)
                labels.append(int(line.split('\\')[1]))
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        label = self.labels[index]
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.labels)


def classes_txt(root, out_path, num_class=None):
    dirs = os.listdir(root)
    if not num_class:
        num_class = len(dirs)

    with open(out_path, 'w') as f:
        end = 0
        if end < num_class - 1:
            dirs.sort()
            dirs = dirs[end:num_class]
            for dir1 in dirs:
                files = os.listdir(os.path.join(root, dir1))
                for file in files:
                    f.write(os.path.join(root, dir1, file) + '\n')
