# fake_news_detection
虚假新闻检测 计挑 数据集为微信公众号 比赛结果不理想，纯试水

# 版本

 * python 3.11.5 
 * torch 1.12.1 gpu

# 模型

BiRNN，TextCNN，具体见[动手学pytorch](https://zh.d2l.ai/index.html)
StrongFND 使用投票机制融合BiRNN，TextCNN，但是效果不是很好

# 文件说明

 * log 文件夹为训练日志
 * models 文件为模型代码和参数
 * run 提交代码，用于预测
 * train 为数据集，其中包括`html`，`image`，`train.csv`，文件太大就不上传了
 * word_vec 保存预训练参数 为微博词向量300d，具体见[Chinese Word Vectors](https://www.jiqizhixin.com/articles/2018-05-15-10)，文件太大不上传
 * 相关文档 提交要求等
 * config.py 保存训练参数
 * data_analysis.py 数据分析,文本预处理
 * dataset.py 数据集构建与划分，词表实现
 * test.py 测试文件
 * train.py 训练文件
 * utils.py 工具函数
 * vocab.py 词表构建



