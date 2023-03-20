# 【模型+代码/保姆级教程】使用Pytorch实现手写汉字识别

## 前言

参考文章：

> 最初参考的两篇：
>
> [【Pytorch】基于CNN手写汉字的识别](https://blog.csdn.net/weixin_44403922/article/details/104451698)
>
> [「Pytorch」CNN实现手写汉字识别（数据集制作，网络搭建，训练验证测试全部代码）](https://blog.csdn.net/qq_31417941/article/details/97915035)
>
> 模型：
>
> [EfficientNetV2网络详解](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test11_efficientnetV2)
>
> 数据集（不必从这里下载，可以看一下它的介绍）：
>
> [CASIA Online and Offline Chinese Handwriting Databases](http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html)

鉴于已经3202年了，GPT4都出来了，网上还是缺乏汉字识别这种“底层”基础神经网络的能让新手**直接上手跑通**的手把手教程，我就斗胆自己写一篇好了。

本文的主要特点：

1. 使用EfficientNetV2模型真正实现3755类汉字识别
2. 项目开源
3. 预训练模型公开
4. 预制数据集，无需处理直接使用



## 数据集

使用中科院制作的手写汉字数据集，链接直达官网，所以我这里不多介绍，只有满腔敬意。

上面参考的博客可能要你自己下载之后按照它的办法再预处理一下，但是在这个环节出现问题的朋友挺多，本着保姆级教程教程的原则，我把预处理的数据已经传到[北航云盘](https://bhpan.buaa.edu.cn:443/link/C2E69919DF187EB23C26653A0483D34D)了，速度应该比百度网盘快吧，大概…

预训练模型已经上传了（后面有链接），但是如果想自己训一下，就需要下载这个数据集，解压到项目结构里的data文件夹如下所示

data文件夹和log文件夹需要自己建。

## 项目结构

完整源代码：【[项目源码](https://github.com/Katock-Cricket/Chinese_Character_Rec)】

![image-20230320160715262](C:/Users/78728/AppData/Roaming/Typora/typora-user-images/image-20230320160715262.png)

**目录结构**

重点注意data文件夹的结构，不要把数据集放错位置了或者多嵌套了文件夹

> ├─Chinese_Character_Rec
> │   ├─asserts
> │   │   ├─*.png
> │   ├─char_dict
> │   ├─Data.py
> │   ├─EfficientNetV2
> │   │   ├─demo.py
> │   │   ├─EffNetV2.py
> │   │   ├─Evaluate.py
> │   │   ├─model.py
> │   │   └─Train.py
> │   ├─Utils.py
> │   ├─VGG19
> │   │   ├─demo.py
> │   │   ├─Evaluate.py
> │   │   ├─model.py
> │   │   ├─Train.py
> │   │   └─VGG19.py
> ├─data
> │   ├─test
> │   │   ├─00000
> │   │   ├─00001
> │   │   ├─00002
> │   │   ├─00003
> │    |    └─...
> │   ├─test.txt
> │   ├─train
> │   │   ├─00000
> │   │   ├─00001
> │   │   ├─00002
> │   │   ├─00003
> |    |   └─ ...
> │   └─train.txt
> ├─log
> │   ├─log1.pth
> │   └─…
> └─README.md



## 神经网络模型

预训练模型[参数链接](https://bhpan.buaa.edu.cn:443/link/719865B23D5DA304FC491A0A65FE24A3)（包含vgg19和efficientnetv2）

请将.pth文件**重命名**为log+数字.pth的格式，例如`log1.pth`，放入log文件夹。方便识别和retrain。

### VGG19

这里先后用了两种神经网络，我先用VGG19试了一下，分类前1000种汉字。训得有点慢，主要还是这模型有点老了，参数量也不小。而且要改到3755类的话还用原参数的话就很难收敛，也不知道该怎么调参数了，估计调好了也会规模很大，所以这里VGG19模型的版本只能分类1000种，就是数据集的前1000种（准确率>92%）。

### EfficientNetV2

这个模型很不错，主要是卷积层的部分非常有效，参数量也很少。直接用small版本去分类3755个汉字，半小时就收敛得差不多了。所以本文用来实现3755类汉字的模型就是EfficientNetV2（准确率>89%)，后面的教程都是基于这个，VGG19就不管了，在源码里感兴趣的自己看吧。



<u>*以下代码不用自己写，前面已经给出完整源代码了，下面的教程是结合源码的讲解而已。*</u>

## 运行环境

<u>**显存>=4G**</u>（与batchSize有关，batchSize=512时显存占用4.8G；如果是256或者128，应该会低于4G，虽然会导致训得慢一点)

<u>**内存>=16G**</u>（训练时不太占内存，但是刚开始加载的时候会突然占一下，如果小于16G还是怕爆）

如果你没有安装过Pytorch，啊，我也不知道怎么办，你要不就看看安装Pytorch的教程吧。（总体步骤是，有一个不太老的N卡，先去驱动里看看cuda版本，安装合适的CUDA，然后根据CUDA版本去pytorch.org找到合适的安装指令，然后在本地pip install）

以下是项目运行环境，我是3060 6G，CUDA版本11.6

这个约等号不用在意，可以都安装最新版本，反正我这里应该没用什么特殊的API

```shell
torch~=1.12.1+cu116
torchvision~=0.13.1+cu116
Pillow~=9.3.0
```



## 数据集准备

首先定义`classes_txt`方法在Utils.py中（不是我写的，是CSDN那两篇博客的，下同）：

生成每张图片的路径，存储到train.txt或test.txt。方便训练或评估时读取数据

```python
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
```

定义Dataset类，用于制作数据集，为每个图片加上对应的标签，即图片所在文件夹的代号

```python
class MyDataset(Dataset):
    def __init__(self, txt_path, num_class, transforms=None):
        super(MyDataset, self).__init__()
        images = []
        labels = []
        with open(txt_path, 'r') as f:
            for line in f:
                if int(line.split('\\')[1]) >= num_class:
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
```



## 入口

我把各种超参都放在了args里方便改，请根据实际情况自行调整。这套defaults就是我训练这个模型时使用的超参，图片size默认32是因为我显存太小辣！！但是数据集给的图片大小普遍不超过64，如果想训得更精确，可以试试64*64的大小。

如果你训练时爆mem，请调小batch_size，试试256，128，64，32

```python
parser = argparse.ArgumentParser(description='EfficientNetV2 arguments')
parser.add_argument('--mode', dest='mode', type=str, default='demo', help='Mode of net')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='Epoch number of training')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=512, help='Value of batch size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='Value of lr')
parser.add_argument('--img_size', dest='img_size', type=int, default=32, help='reSize of input image')
parser.add_argument('--data_root', dest='data_root', type=str, default='../../data/', help='Path to data')
parser.add_argument('--log_root', dest='log_root', type=str, default='../../log/', help='Path to model.pth')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=3755, help='Classes of character')
parser.add_argument('--demo_img', dest='demo_img', type=str, default='../asserts/fo2.png', help='Path to demo image')
args = parser.parse_args()


if __name__ == '__main__':
    if not os.path.exists(args.data_root + 'train.txt'):
        classes_txt(args.data_root + 'train', args.data_root + 'train.txt', args.num_classes)
    if not os.path.exists(args.data_root + 'test.txt'):
        classes_txt(args.data_root + 'test', args.data_root + 'test.txt', args.num_classes)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'demo':
        demo(args)
    else:
        print('Unknown mode')
```



## 训练

在前面CSDN博客的基础上，增加了`lr_scheduler`自行调整学习率（如果连续2个epoch无改进，就调小lr到一半），增加了连续训练的功能：

先在log文件夹下寻找是否存在参数文件，如果没有，就认为是初次训练；如果有，就找到后缀数字最大的log.pth，在这个基础上继续训练，并且每训练完一个epoch，就保存最新的log.pth，代号是上一次的+1。这样可以多次训练，防止训练过程中出错，参数文件损坏前功尽弃。

其中`has_log_file`和`find_max_log`在Utils.py中有定义。

```python
def train(args):
    print("===Train EffNetV2===")
    transform = transforms.Compose(
        [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         transforms.ColorJitter()])

    train_set = MyDataset(args.data_root + 'train.txt', num_class=args.num_classes, transforms=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    device = torch.device('cuda:0')
    model = efficientnetv2_s(num_classes=args.num_classes)
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    print("load model...")

    if has_log_file(args.log_root):
        max_log = find_max_log(args.log_root)
        print("continue training with " + max_log + "...")
        checkpoint = torch.load(max_log)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch'] + 1
    else:
        print("train for the first time...")
        loss = 0.0
        epoch = 0

    while epoch < args.epoch:
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
                   args.log_root + 'log' + str(epoch) + '.pth')
        print('Saved')
        epoch += 1

    print('Finish training')
```



## 评估

没什么好说的，就是跑测试集，算总体准确率。但是有一点不完善，就是看不到每一个类具体的准确率。我的预训练模型其实感觉有几类是过拟合的，但是我懒得调整了。

```python
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
    test_loader = DataLoader(MyDataset(args.data_root + 'test.txt', num_class=args.num_classes, transforms=transform),batch_size=args.batch_size, shuffle=False)
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
```



## 推理

输入文字图片，输出识别结果：

其中char_dict就是每个汉字在数据集里的代号对应的gb2312编码，这个模型的输出结果是它在数据集里的代号，所以要查这个char_dict来获取它对应的汉字。

```python
def demo(args):
    print('==Demo EfficientNetV2===')
    print('Input Image: ', args.demo_img)
    transform = transforms.Compose(
        [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = Image.open(args.demo_img)
    img = transform(img)
    img = img.unsqueeze(0)
    model = efficientnetv2_s(num_classes=args.num_classes)
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
```

例如输入图片为：![image-20230320171619381](C:/Users/78728/AppData/Roaming/Typora/typora-user-images/image-20230320171619381.png)

程序运行结果：![image-20230320171654814](C:/Users/78728/AppData/Roaming/Typora/typora-user-images/image-20230320171654814.png)



## 其他说明

这个模型我正在尝试移植到安卓应用，因为Pytorch有一套Pytorch for Android，但是现在遇到一个问题，它的`bitmap2Tensor`函数内部实现与Pytorch的`toTensor()+Normalize()`不一样，导致输入相同的图片，转出来的张量是不一样的，比如我输入的图片是白底黑字，白底的部分输出一样，但是黑色的部分的数值出现了偏移，我用的是同一套归一化参数，不知道这是为什么。然后这个张量的差异就导致安卓端表现很不好，目前正在寻找解决办法。

另外，这个模型对于太细太黑的字体，准确度貌似不是很好，可能还是有点过拟合了。建议输入的图片与数据集的风格靠拢，黑色尽量浅一点，线不要太细。

