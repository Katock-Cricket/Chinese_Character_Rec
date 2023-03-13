import torch

from Arguments import Args
from vgg19 import VGG19

# pytorch环境中
model_pth = Args.log_root + Args.log_name  # 模型的参数文件
mobile_pt = 'model.pt'  # 将模型保存为Android可以调用的文件
model = VGG19()
checkpoint = torch.load(model_pth)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # 模型设为评估模式
device = torch.device('cpu')
model.to(device)
# 1张3通道224*224的图片
input_tensor = torch.rand(1, 3, 32, 32)  # 设定输入数据格式
mobile = torch.jit.trace(model, input_tensor)  # 模型转化
mobile.save(mobile_pt)  # 保存文件
# mobile = torch.jit.trace(model, input_tensor)  # 模型转化
# opt_model = optimize_for_mobile(mobile)
# opt_model._save_for_lite_interpreter("model2.pt")
