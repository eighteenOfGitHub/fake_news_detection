from torch import nn
import torch 
from torch import Tensor

from models.TextCNN import TextCNN
from models.BiRNN import BiRNN


class StrongFND(nn.Module):
    class Voting(nn.Module):
        def __init__(self, **kwargs):
            super(StrongFND.Voting, self).__init__(**kwargs)
            self.weight = nn.Parameter(Tensor([0.5, 0.5]))
            self.ones = nn.Parameter(torch.ones(2))
        def forward(self, out1, out2):
            return out1*self.weight + out2*(self.ones - self.weight)

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 kernel_sizes, num_channels, **kwargs):
        super(StrongFND, self).__init__(**kwargs)
        self.model_1 = BiRNN(vocab_size, embed_size, num_hiddens, num_layers)
        self.model_2 = TextCNN(vocab_size, embed_size, kernel_sizes, num_channels)
        self.voting = self.Voting()

    def forward(self, inputs):
        out1 = self.model_1(inputs)
        out2 = self.model_2(inputs)
        return self.voting(out1, out2)
