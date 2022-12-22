import time
import torch
from vgg19 import VGG19

def proc_time(b_sz, model, n_iter=500):
    # 模型输入部分
    x = torch.rand(b_sz, 3, 32, 32).cuda()  # <----- 在这里设置输入的形状

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        model(x)  # <---- 模型输入
    torch.cuda.synchronize()
    end = time.time() - start
    throughput = b_sz * n_iter / end
    print(f"Batch: {b_sz} \t {throughput} samples/sec")
    return (b_sz, throughput,)


if __name__ == '__main__':
    model = VGG19()
    device = torch.device('cuda:0')
    model.to(device)
    for b_sz in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        proc_time(b_sz, model)
