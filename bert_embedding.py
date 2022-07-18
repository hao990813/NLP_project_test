import tensorflow as tf
import random
#使用tensorflow上的预训练模型bert
import tensorflow_hub as hub
import pandas as pd
import numpy as np
#模型预处理时内部使用到了对应的模块，所以必须导入
import tensorflow_text as text
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split # 进行训练和测试样本的分割(sklearn 0.23.1)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential #(keras2.9.0)
from tensorflow.keras.layers import Activation,Dense, GRU, Embedding, LSTM, Bidirectional,Dropout,BatchNormalization#Dense全连接
#读取之前数据处理保存的训练集与测试集
df_full_data_save = pd.read_csv('df_full_data.csv',encoding='utf_8_sig')
# print(len(df_full_data_save))

#删除datframe中的空值
df_full_data_save.dropna(inplace=True)
# print(len(df_full_data_save))

#对标签进行one-hot编码，转换成神经网络能够接受的格式
ohe = OneHotEncoder()
ohe.fit([[-1],[0],[1]])
senti_label = ohe.transform(df_full_data_save['pos_or_neg'].values.reshape(-1,1)).toarray()

# #将公告转换成768维的向量，利用tensorflow_hub
# #预训练模型
# preprocessing_layer = hub.KerasLayer("https://hub.tensorflow.google.cn/tensorflow/bert_zh_preprocess/3"
#                                        , name='preprocessing')
# encoder_inputs = preprocessing_layer(df_full_data_save['公告标题'].values)
# #bert结果输出
# encoder = hub.KerasLayer("https://hub.tensorflow.google.cn/tensorflow/bert_zh_L-12_H-768_A-12/4"
#                            , trainable=False, name='BERT_encoder')
# outputs = encoder(encoder_inputs)
# news_vec = outputs['sequence_output']
# news_vec = np.array(news_vec)
# np.save('news_vector_seq',news_vec)

#直接读取之前已经通过bert模型训练出来的向量
# news_vec = np.load('news_vector.npy')
news_vec = np.load('news_vector_seq.npy')

#划分测试与与训练集
x_train, x_test, y_train, y_test = train_test_split(news_vec,senti_label,test_size=0.3,random_state=12)
# print(x_train.shape)

#将测试的label从one-hot的格式转成int方便后面classification_report调用
list_test = []
#找到最大值所在的位置
for i in range(len(y_test)):
    list_test.append(np.argmax(y_test[i]))
#对应到具体的-1，0或者1
y_test_list = []
for i in range(len(list_test)):
    if list_test[i] == 0:
        y_test_list.append(-1)
    if list_test[i] == 1:
        y_test_list.append(0)
    if list_test[i] == 2:
        y_test_list.append(1)

#创建一个Sequential模型
model = Sequential()
# #双向LSTM考虑前后
model.add(Bidirectional(LSTM(units=64, return_sequences=False)))#双向LSTM考虑前后词

#加入dropout层避免过拟合
model.add(Dropout(0.5))
#加入全连接层
model.add(Dense(3, activation='softmax'))
#调用compile函数来指定损失函数以及优化器
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 开始训练,validation_split=0.2
history = model.fit(x_train, y_train, validation_split=0.2, epochs=30,batch_size=32)
#打印模型结构
print(model.summary())
#查看模型性能
score = model.evaluate(x_test, y_test)
#只保留两位小数
score = [round(i,2) for i in score]
print(score)

#预测测试及数据
y_pred = model.predict(x_test)
#将预测后的label从one-hot的格式转成int方便后面classification_report调用
list_pred = []
for i in range(len(y_pred)):
    list_pred.append(np.argmax(y_pred[i]))
#对应到具体的-1，0或者1
y_pred_list = []
for i in range(len(list_pred)):
    if list_pred[i] == 0:
        y_pred_list.append(-1)
    if list_pred[i] == 1:
        y_pred_list.append(0)
    if list_pred[i] == 2:
        y_pred_list.append(1)

print(y_test_list)
print(y_pred_list)
print("精确率和召回率为：", classification_report(np.array(y_test_list), np.array(y_pred_list), labels=[-1,0,1], target_names=['消极', '中性','积极']))

#观察最优的val_accuracy是出现在哪一轮
result_word2vec =pd.DataFrame(history.history)
k=max(result_word2vec.val_accuracy)
result_word2vec[result_word2vec.val_accuracy==k]
print(result_word2vec)

#保存模型
model.save('./bert_text_classifier')
# #加载模型
# model = tf.keras.models.load_model('./word2vec_text_classifier')










