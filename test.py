

# 写一个输入为线性层输入为10输出为2的模型
import torch
import torch.nn as nn
from torch import Tensor
from models.BiRNN import BiRNN
from models.StrongFND import StrongFND
from config import *
from utils import *
tensor = Tensor([0.4, 0.6])
print(tensor)


def test():
    net1 = StrongFND(len(vocab), embed_size, num_hiddens, num_layers,
                 kernel_sizes, nums_channels)
    net2 = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
    # net1.load_state_dict(torch.load('./models/StrongFND_right_text.pth'))
    net2.load_state_dict(torch.load('./models/BiRNN_title.pth'), strict=False)
    # net1.eval()
    net2.eval()
    print('ok')

test()
