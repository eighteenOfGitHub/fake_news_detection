import torch
from torch import nn
from d2l import torch as d2l

from models.BIRNN import BiRNN
from dataset import load_data_righttext_db
from utils import try_all_gpus, TokenEmbedding





batch_size = 64
train_iter, test_iter, vocab = load_data_righttext_db(batch_size)
embed_size, num_hiddens, num_layers = 300, 300, 2
devices = try_all_gpus()

net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
net.apply(init_weights)

glove_embedding = TokenEmbedding('sgns.weibo.char')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False

lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")

