# fake_news_detection
虚假新闻检测 计挑 数据集为微信公众号 比赛结果不理想，纯试水

# 版本

 * python 3.11.5 
 * torch 1.12.1 gpu

# 数据处理  

### 数据提取  

通过utf-8编码将html文件中的文字和汉字符号部分提取出来

### 数据清洗  

 * 删除无用的字段，比如“分享留言收藏”等
   
 * 将无法使用的数据筛选掉，比如从thml提取出来的文档包含“此内容因违规无法查看”等内容的

# 模型

BiRNN，TextCNN，具体见[动手学pytorch](https://zh.d2l.ai/index.html)
StrongFND 使用投票机制融合BiRNN，TextCNN，但是效果不是很好

# 思路/策略

因为数据集大部分因为被删除无法查看，所以将数据集分成两类，有内容和无内容的，分别训练一个模型，有内容的数据集为html处理的内容，无内容的使用标题、作者名等信息拼接后作为数据集。预测前将测试集分类，分类后进入不同的模型进行预测。

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



