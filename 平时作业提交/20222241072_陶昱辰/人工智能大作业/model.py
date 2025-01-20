# import mindspore
# import mindspore.nn as nn
# import mindspore.ops.functional as F
# import mindspore.common.dtype as mstype
# from mindspore import Tensor, context
# import numpy as np
#
# # 设置MindSpore的运行模式为图模式或PyNative模式（这里选择图模式）
# context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
#
#
# # CNN网络
# class CNN(nn.Cell):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.model = nn.SequentialCell([
#             # 第一层卷积
#             nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, pad_mode='pad', weight_init='normal',
#                       bias_init='zeros'),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样一半，尺寸变为 512x256
#
#             # 第二层卷积
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, pad_mode='pad', weight_init='normal',
#                       bias_init='zeros'),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样一半，尺寸变为 256x128
#
#             # 第三层卷积
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, pad_mode='pad', weight_init='normal',
#                       bias_init='zeros'),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样一半，尺寸变为 128x64
#
#             # 第四层卷积
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, pad_mode='pad', weight_init='normal',
#                       bias_init='zeros'),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样一半，尺寸变为 64x32
#
#             # 展平
#             nn.Flatten(),
#             # 原图尺寸为 1024x512，通道数 128，经过四次 MaxPool2d 后，尺寸变为 32x16
#             nn.Dense(128 * 64 * 32, 512, weight_init='normal', bias_init='zeros'),
#             nn.ReLU(),
#             # 防止过拟合
#             nn.Dropout(0.2),
#             # 八分类
#             nn.Dense(512, 8, weight_init='normal', bias_init='zeros')
#         ])
#
#     def construct(self, x):
#         return self.model(x)
#
#
# # 测试模型正确性
# if __name__ == '__main__':
#     model_test = CNN()
#     input_data = Tensor(np.ones((32, 1, 1024, 512)), mstype.float32)  # 注意：这里需要导入numpy并使用numpy数组
#     output = model_test(input_data)
#     print(output.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 64 * 32, 512)  # 假设输入尺寸为1024x512
        self.fc2 = nn.Linear(512, 8)  # 八分类

        # Dropout层
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 第一层卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv1(x)))
        # 第二层卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv2(x)))
        # 第三层卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv3(x)))
        # 第四层卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv4(x)))

        # 展平
        x = x.view(x.size(0), -1)  # 扁平化，x.size(0) 是批大小

        # 全连接层 + 激活
        x = F.relu(self.fc1(x))

        # Dropout
        x = self.dropout(x)

        # 输出层
        x = self.fc2(x)

        return x


# 测试模型正确性
if __name__ == '__main__':
    model_test = CNN()
    input_data = torch.ones((32, 1, 1024, 512))  # 注意：这里的输入是一个大小为 (32, 1, 1024, 512) 的Tensor
    output = model_test(input_data)
    print(output.shape)  # 打印输出的shape