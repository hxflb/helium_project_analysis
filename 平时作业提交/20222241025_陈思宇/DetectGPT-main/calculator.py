import torchvision.models
import torch
from thop import profile
from thop import clever_format
# import torchsummary
model = torchvision.models.GPT2PPLV2(pretrained=False)
device = torch.device('cuda')
model.to(device)
myinput = torch.zeros((1, 3, 224, 224)).to(device)
flops, params = profile(model.to(device), inputs=(myinput,))
flops, params = clever_format([flops, params], "%.3f")
print(flops, params)