####################################################################
                #导入项目所需要到的库，版本在括号中显示
####################################################################
import re #导入re(2.2.1)正则化用
import numpy as np #numpy(1.23.1)
import pandas as pd #导入pandas(1.4.3)
import jieba   #导入jieba(0.42.1)中文分词
import sqlalchemy #sqlalchemy（1.4.7）链接股票数据库
import copy
import tensorflow
from sklearn.model_selection import train_test_split # 进行训练和测试样本的分割(sklearn 0.23.1)
from sklearn.preprocessing import OneHotEncoder#(sklearn 0.23.1)
from sklearn.metrics import classification_report#(sklearn 0.23.1)
from cnsenti import Sentiment #(cnsenti0.0.7)
from keras.optimizers import Adam#(keras2.9.0)
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau#(keras2.9.0)
from keras.models import Sequential #(keras2.9.0)
from keras.layers import Activation,Dense, GRU, Embedding, LSTM, Bidirectional,Dropout,BatchNormalization#Dense全连接 #(keras2.9.0)
from gensim.models import KeyedVectors ## gensim(3.8.3)用来加载预训练word vector
from tensorflow.keras.preprocessing.sequence import pad_sequences #进行填充和剪枝用的（2.9.1tensorflow）

#######################################################################
                #设置输出的显示格式为显示所有数据行列
######################################################################
#显示所有列
pd.set_option('display.max_columns',None)
#显示所有行
pd.set_option('display.max_rows',None)
#设置value的显示长度
pd.set_option('max_colwidth',100)
#设置1000列时才换行
pd.set_option('display.width',1000)

#######################################################################
                #读取数据并进行数据处理
######################################################################
#使用pandas读入csv数据
df_news_data = pd.read_csv("./data/news_5_22_to_5_17_test.csv",encoding='gbk')
# print(df_news_data.head(5))
#将日期转换为datetime格式
df_news_data['公告日期'] = pd.to_datetime(df_news_data['公告日期'],infer_datetime_format=True)
# print(df_news_data['公告标题'].head())
# print(type(df_news_data['公告日期'][0]))

#补全获取的股票代码
df_news_data['代码'] = df_news_data['代码'].astype('str') #将数据类型转换为文本
df_news_data['代码']  = df_news_data['代码'].str.zfill(6) #用的时候必须加上.str前缀
# print(df_news_data.head(5))
# print(type(df_news_data['公告类型'][0]))

#创建一个新的dataframe来保存jieba分词以后的中文，以及对应的索引向量
df_load = pd.DataFrame()
list_fenci_save = []

#读取哈工大停用表
stopword_list = [k.strip() for k in open('./files/哈工大停用词表.txt', encoding='gbk').readlines() if k.strip() != '']

#
senti = Sentiment(pos='./files/formal_pos.txt',  # 正面词典txt文件相对路径
                  neg='./files/formal_neg.txt',  # 负面词典txt文件相对路径
                  merge=False,  # 是否将cnsenti自带词典和用户导入的自定义词典融合
                  encoding='utf-8')  # 两txt均为utf-8编码

#对公告标题进行分词并进行标点符号的处理并使用jieba分词切割
for i in range(len(df_news_data['公告标题'])):
    #使用re正则化只保留新闻中的中文字符
    df_news_data._set_value(i,'公告标题',[k for k in (re.sub('[^\u4e00-\u9fa5]+|公告', '', df_news_data['公告标题'][i])) if k not in stopword_list])
    #进行情感打分
    df_news_data._set_value(i,'公告标题',''.join(df_news_data['公告标题'][i]))
    #每次循环将中文分词结果储存（深拷贝）
    list_fenci_save.append(copy.deepcopy(df_news_data['公告标题'][i]))
    #加上股票代码的后缀
    if df_news_data['代码'][i][0] == "6":
        df_news_data._set_value(i,'代码',df_news_data['代码'][i] + ".SH")
    elif df_news_data['代码'][i][0] == "0":
        df_news_data._set_value(i,'代码',df_news_data['代码'][i] + ".SZ")
    elif df_news_data['代码'][i][0] == "3":
        df_news_data._set_value(i, '代码', df_news_data['代码'][i] + ".SZ")

# print(df_news_data['公告标题'].head())
#保存分词以及对应的中文索引结果到df_load中
df_load['分词'] = list_fenci_save
df_load['分词索引化'] = df_news_data['公告标题']

#只研究沪深股,直接删除北交所的数据
# 利用enumerate进行遍历，将含有数字3的列放入cols中
rows = [x for i, x in enumerate(df_news_data['代码']) if df_news_data['代码'][i][0] == "8"]
#利用~在dataframe中进行反选不是北交所的
df_news_data =  df_news_data[~df_news_data['代码'].isin(rows)]
#重新索引
df_news_data = df_news_data.reset_index(drop=True)
# print(df_news_comine['公告标题'][146][1])
# print(df_news_data)

#进行类型的转换
df_news_data['公告标题'] = df_news_data['公告标题'].astype('str')
#根据公告日期和代码对df进行聚合操作
gb = df_news_data.groupby(by=['公告日期','代码'])
# 将df_news_data中相同日期相同股票代码的公告标题进行合并，将标题存在list中保存在df
df_news_comine = gb['公告标题'].unique()
#重新索引
df_news_comine = df_news_comine.reset_index()
# print(df_news_comine.head(20))

#统计出每一条公告中出现的pos词与neg次的个数
count_pos = []
count_neg = []
pos_temp = 0
neg_temp = 0
for i in range(len(df_news_comine['公告标题'])):
    for j in range(len(df_news_comine['公告标题'][i])):
        pos_temp = pos_temp + senti.sentiment_count(df_news_comine['公告标题'][i][j])['pos']
        neg_temp = neg_temp + senti.sentiment_count(df_news_comine['公告标题'][i][j])['neg']
    count_pos.append(pos_temp)
    count_neg.append(neg_temp)
    pos_temp = 0
    neg_temp = 0

#将统计出来的个数存入df中去
df_news_comine['pos'] = count_pos
df_news_comine['neg'] = count_neg
# print(len(df_news_comine))





