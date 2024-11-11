import torch
import os
import string
import pandas as pd 
from bs4 import BeautifulSoup
# from tqdm import tqdm

from utils import *
from models.BiRNN import BiRNN
from models.StrongFND import StrongFND

dirty_key_words = ['此内容因违规无法查看', '该内容已被发布者删除', '此账号已被屏蔽内容无法查看',
                   '此账号已自主注销', '原账号迁移时未将文章素材同步至新账号','参数错误']
dirty_words = ["：，。视频小程序赞，轻点两下取消赞在看，轻点两下取消在看",
               "分享留言收藏",
               "向上滑动看下一个知道了微信扫一扫使用小程序取消允许取消允许分析",
               "预览时标签不可点微信扫一扫关注该公众号继续滑动看下一个轻触阅读"
               ]

embed_size, num_hiddens, num_layers = 300, 300, 2
kernel_sizes, nums_channels =  [3, 4, 5], [100, 100, 100]

def get_data(testa_csv_path, testa_html_dir):

    """得到公众号名、标题"""

    # print('leading title, official account name...')
    testa =  pd.read_csv(testa_csv_path)
    titles = [str(t) for t in testa["Title"].to_list()]
    official_account_names = [ str(o) for o in testa["Ofiicial Account Name"].to_list()]
    # print(len(titles), len(official_account_names))

    """得到文本"""

    # 读取html文件

    # print('leading html...')
    html_list = []
    for i in range(len(testa)):
        id = testa.loc[i, "id"]
        html_i = os.path.join(testa_html_dir, "%s.html"%id)
        html = open(html_i, "r", encoding='utf-8').read().strip()
        html_list.append(html)
    # print(len(html_list))

    # 将html中的中文内容提取出来

    # print('getting text...')
    number = [n for n in string.digits]
    chinese_symbol = ['。', '，', '！', '？', '；', '：', '“', '”', '‘', '’', '（', '）', '《', '》', '、', '…', '—', '【', '】']
    texts = []
    for html in html_list:
        soup = BeautifulSoup(html)
        chinese_text = ''.join(soup.stripped_strings)
        chinese_text = [char for char in chinese_text if '\u4e00' <= char <= '\u9fff' or char in chinese_symbol or char in number]
        texts.append(''.join(chinese_text))
    # print(len(texts))

    # 删除无用数据

    # print('clearing text ...')
    for index, _ in enumerate(texts):
        for dirty_word in dirty_words:
            texts[index] = texts[index].replace(dirty_word, '')

    return texts, titles, official_account_names

def  val(texts, titles, official_account_names, run_dir): # 虚假为1
    """
    text: list                          type of element: str
    title: list                         type of element: str
    official_account_names: list        type of element: str
    """

    # 数据、模型准备
    # print('loading data and model...')
    result = []
    texts1 = texts
    texts2 = [official_account_names[i]+'，'+titles[i] for i in range(len(titles))]
    vocab1, tokens1 = init_vocab(texts1, 10)
    vocab2, tokens2 = init_vocab(texts2, 2)
    net1 = StrongFND(len(vocab1), embed_size, num_hiddens, num_layers,
                 kernel_sizes, nums_channels)
    net2 = BiRNN(len(vocab2), embed_size, num_hiddens, num_layers)
    net1.load_state_dict(torch.load(os.path.join(run_dir,"models", "BiRNN_right_text.pth")), strict=False)
    net2.load_state_dict(torch.load(os.path.join(run_dir,"models", "BiRNN_title.pth")), strict=False)
    net1.eval()
    net2.eval()

    # 预测
    print('predicting...')
    usable = 0
    unsuable = 0
    for i in range(len(texts)):
    # for i in tqdm(range(len(texts))):
        # 模型选择、预测内容选择、词表选择
        if texts[i] == '':
            net = net2
            token = tokens2[i]
            vocab = vocab2
            num_steps = 30
            unsuable += 1
        else:
            for del_key_word in dirty_key_words:
                if del_key_word in texts[i]:
                    net = net2
                    token = tokens2[i]
                    vocab = vocab2
                    num_steps = 30
                    unsuable += 1
                    break
            else:
                net = net1
                token = tokens1[i]
                vocab = vocab1
                num_steps = 900
                usable += 1
        # 预测
        net.to(try_gpu())
        feature = torch.tensor([truncate_pad(
            vocab[token], num_steps, vocab['<pad>'])], device=try_gpu())
        label = torch.argmax(net(feature), dim=1)
        result.append(label.item())
    # print(f'usable: {usable}, unsuable: {unsuable}')
    return result

def model_out_cxy(testa_csv_path, testa_html_dir, run_dir):
    texts, titles, official_account_names = get_data(testa_csv_path, testa_html_dir)
    result = val(texts, titles, official_account_names, run_dir)
    return result

def main():
    ################ 需要改动的地方 ################
    run_py = os.path.abspath(__file__)
    model_dir = os.path.dirname(run_py)
    # print(model_dir)

    to_pred_dir = 'E:\\_codes\\fake_news_detection'
    to_pred_dir = os.path.abspath(to_pred_dir)
    testa_csv_path = os.path.join(to_pred_dir, "train", "train.csv")
    testa_html_dir = os.path.join(to_pred_dir, "train", "html")

    ################################################
    result = model_out_cxy(testa_csv_path, testa_html_dir, model_dir)
    # print(result[0], result[1], result[2])
    # print(len(result))
    testa =  pd.read_csv(testa_csv_path)
    tabels = testa["label"].to_list()
    right_count = 0
    for y_hat, y in zip(result, tabels):
        if y_hat == y:  
            right_count += 1
    print('right_count: ', right_count)
    print('right_rate: ', right_count/len(result))

if __name__ == '__main__':
    main()