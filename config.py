import torch
from torch import nn
from d2l import torch as d2l

from models.BiRNN import BiRNN
from models.TextCNN import TextCNN
from models.StrongFND import StrongFND
from dataset import load_data_db
from utils import try_all_gpus, TokenEmbedding

batch_size = 64
devices = try_all_gpus()

db_name = "right_text" # right_text title
net_name = "BiRNN" # "TextCNN" "BiRNN" "StrongFND"

train_iter, test_iter, vocab = load_data_db(db_name, batch_size)
embed_size, num_hiddens, num_layers = 300, 300, 4
kernel_sizes, nums_channels = [3, 4, 5], [100, 100, 100]

if net_name == "BiRNN":
    net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.LSTM:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(init_weights)

    # print('loding embedding...')
    # glove_embedding = TokenEmbedding('sgns.weibo.char')
    # embeds = glove_embedding[vocab.idx_to_token]
    # net.embedding.weight.data.copy_(embeds)
    # net.embedding.weight.requires_grad = False
    # print('embedding loaded')

elif net_name == "TextCNN":

    net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

    def init_weights(m):
        if type(m) in (nn.Linear, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    # print('loding embedding...')
    # glove_embedding = TokenEmbedding('sgns.weibo.char')
    # embeds = glove_embedding[vocab.idx_to_token]
    # net.embedding.weight.data.copy_(embeds)
    # net.constant_embedding.weight.data.copy_(embeds)
    # net.constant_embedding.weight.requires_grad = False
    # print('embedding loaded')

elif net_name == "StrongFND":

    net = StrongFND(len(vocab), embed_size, num_hiddens, num_layers,
                 kernel_sizes, nums_channels)
    
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.LSTM:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
        if type(m) == nn.Conv1d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    # print('loding embedding...') 
    # glove_embedding = TokenEmbedding('sgns.weibo.char')
    # embeds = glove_embedding[vocab.idx_to_token]
    # net.model_1.embedding.weight.data.copy_(embeds)
    # net.model_1.embedding.weight.requires_grad = False
    # net.model_2.embedding.weight.data.copy_(embeds)
    # net.model_2.constant_embedding.weight.data.copy_(embeds)
    # net.model_2.constant_embedding.weight.requires_grad = False
    # print('embedding loaded')

lr, num_epochs = 0.001, 10
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
# torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2, eta_min=0, last_epoch=-1)




