import os
import torch
import pandas as pd 
import random
import jieba
import matplotlib.pyplot as plt

from vocab import Vocab


def read_db(db_name):

    if db_name == "right_text":
        text = []
        data_dir = "./train/right_text"
        labels = pd.read_csv('./train/usable_train.csv')['label'].to_list()
        for file in os.listdir(data_dir):
            with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                news = f.read().replace('\n', '')   
                text.append(news)
    elif db_name == "title":
        train = pd.read_csv('./train/train.csv')
        labels = train['label'].to_list()
        text = [str(o)+'，'+str(t).replace(' ', '') for o,t in zip(train['Ofiicial Account Name'].to_list(), train['Title'].to_list())]


    # 打乱数据集，80%为训练集，20%为测试集
    
    len_train = int(len(text) * 0.8) 
    random.seed(1)
    ids = [i for i in range(len(text))]
    random.shuffle(ids)

    train_text = [text[i] for i in ids[:len_train]]
    test_text = [text[i] for i in ids[len_train:]]
    train_label = [labels[i] for i in ids[:len_train]]
    test_label = [labels[i] for i in ids[len_train:]]
    return (train_text, train_label), (test_text, test_label)

def tokenize(lines):
    tokenize_lines = [[i for i in jieba.cut(line)] for line in lines]
    return tokenize_lines

def truncate_pad(line, num_steps, padding_token): # 
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

def load_array(data_arrays, batch_size, is_train=True):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

def load_data_db(db_name, batch_size):
    """返回数据迭代器和数据集的词表"""
    if db_name == "right_text":
        num_steps = 900
        min_freq = 10
    elif db_name == "title":
        num_steps = 30
        min_freq = 2

    train_data, test_data = read_db(db_name)

    train_tokens = tokenize(train_data[0])
    test_tokens = tokenize(test_data[0])

    # plt.xlabel('# tokens per review')
    # plt.ylabel('count')
    # plt.hist([len(line) for line in train_tokens]) # , bins=range(0, 1000, 50)
    # plt.show()

    
    vocab = Vocab(train_tokens, min_freq)
    train_features = torch.tensor([truncate_pad( # 截断或填充
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab

def main():
    train_iter, test_iter, vocab = load_data_db('title', 64)
    for X, y in train_iter:
        print('X:', X.shape, ', y:', y.shape)
        break
    print('小批量数目：', len(train_iter))

if __name__ == '__main__':
    main()
