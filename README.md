项目逻辑：

项目旨在通过对新闻文本进行情感分类，使文本公告转换为情感特征，结合其他指标对后一天的收盘价数据进行预测。

首先对新闻公告文本进行处理，根据word2vec/bert预训练方法将文本进行情感分类（积极，消极，中立），将训练好的情感标签结合股票当天的开盘价，收盘价，最高价，最低价，成交量，时间，股票代码这几个特征进行机器学习/深度学习预测第二天股票收盘价的涨跌情况（涨，跌，平），进行对比实验看哪种模型具有更好的预测效果，并看情感特征是否会增加预测的精度。

详细步骤介绍：

原始数据：

原始数据是从东方财富网得到的公告，网站为 https://data.eastmoney.com/notices/ ，数据日期为2022/5/17-2022/5/22，4000+条数据，数据格式如下
![image](https://github.com/hao990813/NLP_project_test/blob/master/60ac92c8e2ebffac176032835055f1d.png)

数据预处理阶段：

data_with_sentiment文件首先读取储存在data文件夹下的原始数据，引入哈工大停用词表，利用正则表达式只保留公告中不在停用词表中的汉字，另外去除‘公告’这个词。将同一天同一支股票的
公告信息进行合并，对股票代码进行补全，删除北交所股票。引入了中国金融情绪词典，详细介绍： http://www.doc88.com/p-69199879282388.html， files文件夹下包含着pos与neg的txt，
txt文件中均为中文的积极/消极情绪词，统计出该天公告的词在pos中出现的次数与在neg中出现的次数，data_with_sentiment结束后dataframe的格式如图所示,数据量为1700+。
![image](https://github.com/hao990813/NLP_project_test/blob/master/5ffcd6642edc78183049133bbb05b0e.png)

data_merge文件首先进行与mhtz数据库的链接，找到5.17-5.22号之间每个日期所对应‘当天’的交易日与‘后一个’的交易日，涉及到周末，节假日等，通过数据库找交易日，我们将日期大于等于这个日期的最近的交易日作为“当天”，离“当天”之后最近的一个交易日当作“后一个”交易日。

举例：2022/5/20为周五，则对应的‘当天’为2022/5/20，后一个”交易日为2022/5/23，2022/5/21为周六，则对应的‘当天’为2022/5/23，后一个”交易日为2022/5/24，2022/5/22为周天，则对应的‘当天’为2022/5/23，后一个”交易日为2022/5/24。

找到对应的日期，可以从数据库中找出开盘价，收盘价，最高价，最低价，成交量的数据，将得到的dataframe与data_with_sentiment得到的新闻数据dataframe进行merge操作，合并后的dataframe包含着“当天”与“后一个”交易日的收盘价数据，将这两列进行相减可以得出后一个交易日个股的涨跌情况。在本项目中我们不以绝对的涨跌作为标签，为了避免某天是绝对牛/熊市的状态。将得到的涨跌根据全部样本的个股日频收益率（按收盘价价计算）上下三分之一分位数作为阈值，将样本划分为上涨、震荡、下跌三个类别（1，0，-1）。对于情绪词删除积极/消极词均未出现过的句子，不带情感因子不在研究范围内，依据积极情绪词的次数是否大于消极情绪词的词数来进行情感的三分类，据统计目前本项目的情感分类比为积极：中立：消极= 689：121：183。

文本嵌入阶段：

本实验采取了两种文本嵌入方法：word2vec与bert，对于已有数据来说两者效果接近，均能达到85%以上的情感分类结果。
word2vec_embedding文件实现了word2vec，利用北京师范大学和中国人民大学研究者开源的中文预训练词向量 Chinese-Word-Vectors，将切分后的每个词语转化为 300 维的向量。对情感（-1，0，1）进行one-hot编码作为标签，对分词后的公告向量进行填充和剪枝来实现长度一致作为输入。keras提供了Embedding层，结合lstm模型对文本的情感进行深度学习，网络结构以及最后三分类预测的结果如图所示
![image](https://github.com/hao990813/NLP_project_test/blob/master/3fff915bab5fe2797278ee4a397ff0a.png)

bert_embedding文件使用TensorFlow Hub上的预训练模型BERT，针对中文的预训练模型v4，以及配套的preprocess模型，利用输出的sequence_outputt作为中文嵌入向量进行下游工作，将每条新闻公告输入都转化为768维的向量。为了比较两种文本嵌入模型的准确度，将得到的向量同样放入lstm模型中，网络结构以及最后三分类预测的结果如图所示
![image](https://github.com/hao990813/NLP_project_test/blob/master/9b5d4b5929f422f6d971177ae56029f.png)

涨跌预测阶段：

在此阶段采用对比实验的方式来研究情感因子是否能提高股票涨跌预测的精确度，以及在已有数据下神魔模型对股票涨跌的预测效果最好。有四个文件分别为lstm_pred_with_senti,lstm_pred_without_senti,ml_pred_with_senti,ml_pred_without_senti。其中前两个文件都是采用lstm模型对股票涨跌进行预测，采用的lstm种类有普通单层lstm,堆叠lstm,ConvLSTM2D,cnn*lstm,双向lstm五种，后两个文件是通过机器学习的方法对股票涨跌进行预测采用的机器学习模型为LDA线性判别，knn，SVM，决策树，随机森林五种。

第一个和第三个文件将情感作为其中一个特征，第二个和第四个文件则是对比试验没有将情感作为特征，输入的特征为股票当天的开盘价，收盘价，最高价，最低价，成交量，时间，股票代码，预测的是第二天股票收盘价的涨跌平情况（三分类）。准确度预测结果如下图：
![image](https://github.com/hao990813/NLP_project_test/blob/master/135a0c87855c60ea21f2bff2107c27a.png)

结论与总结：

深度学习算法普遍上比传统的机器学习效果要更好，融入的情感特征并未在很大程度上提高股票预测的精度，猜测可能是由于训练样本太少的缘故导致，天数跨度只有五天无法进行很好的预测，
下一步准备扩大数据样本，并且不是单单利用当天的值去预测下一天，可以设置时间戳，利用前n天的数据去进行预测，参考https://blog.csdn.net/IYXUAN/article/details/118530843?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165692059616782246477785%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&amp;amp;request_id=165692059616782246477785&amp;amp;biz_id=0&amp;amp;utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-118530843-null-null.142%5ev30%5econtrol,185%5ev2%5etag_show&amp;amp;utm_term=%E6%83%85%E6%84%9F%E6%A0%87%E7%AD%BE%E5%AF%B9%E8%82%A1%E7%A5%A8%E6%B6%A8%E8%B7%8C%E9%A2%84%E6%B5%8B%E2%80%98&amp;amp;spm=1018.2226.3001.4187


问题：

1.新闻公告标题包含的情感特征不是很明显，可以尝试读取整个公告不仅仅是标题

2.数据量太小时间跨度太短

3.很多研究都基于个股，对个股公告/股评进行研究来预测个股的涨跌，而本项目是针对出现在公告中的所有股票，所以在构建特征的时候加上了公告的时间与股票代码，这样子是否会影响到模型的效果。

参考资料：
Development of a Stock Price Prediction Framework for
Intelligent Media and Technical Analysis

















