import os
import torch
from torch import nn
from d2l import torch as d2l



def read_imdb(data_dir):
    train_data, train_label = [], []
    test_data, test_label = [], []
    
    for label in ('pos', 'neg'):
        for file in os.listdir(data_dir):
            with open(os.path.join(data_dir, file), 'r') as f:
                news = f.read().decode('utf-8').replace('\n', '')
                data.append(news)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

def load_data_imdb(batch_size, num_steps=500):
    """返回数据迭代器和数据集的词表"""
    data_dir = "./train/right_text"
    train_data, test_data = read_imdb(data_dir) 
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab

