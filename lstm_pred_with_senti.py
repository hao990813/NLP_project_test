import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split # 进行训练和测试样本的分割(sklearn 0.23.1)
from keras.models import Sequential #(keras2.9.0)
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Activation,BatchNormalization,MaxPooling1D,Conv1D,Dense,MaxPooling2D, \
    GRU, Embedding, LSTM, Bidirectional,Dropout,BatchNormalization,ConvLSTM2D,Flatten#Dense全连接
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

#读取之前数据处理保存的训练集与测试集
df_full_data_save = pd.read_csv('df_full_data.csv',encoding='utf_8_sig')
# print(len(df_full_data_save))
#删除datframe中的空值
df_full_data_save.dropna(inplace=True)
# print(df_full_data_save.head(20))
# print(len(df_full_data_save))

df_pred = df_full_data_save[['up_or_down','current_close','current_open','current_high',
                             'current_low','current_vol','pos_or_neg','公告日期','代码']]

# #将股票代码转换成数字，作为机器学习的特征
temp_list = df_full_data_save['代码'].tolist()
temp_news = df_pred['公告日期'].tolist()
# print(df_pred['公告日期'][0])
for i in range(len(df_pred)):
    pos = temp_list[i].rfind('.')
    str_temp = ''.join(re.findall('[0-9]',temp_news[i]))
    df_pred.iloc[i,df_pred.columns.get_indexer(['代码'])] = int(temp_list[i][:pos])
    df_pred.iloc[i, df_pred.columns.get_indexer(['公告日期'])] = int(str_temp)
# #


df_pred.columns = [['up_or_down','current_close','current_open','current_high',
                             'current_low','current_vol','pos_or_neg','date','code']]
# print(df_pred.head())

#将公告日期也转换为数字


#归一化操作
transfer = MinMaxScaler()
pred_arr = transfer.fit_transform(df_pred)
# print(pred_arr)

#将标签进行one-hot编码
up_down_pred= np.unique(pred_arr[:,0])
ohe = OneHotEncoder()
ohe.fit([[up_down_pred[0]],[up_down_pred[1]],[up_down_pred[2]]])
up_down_label = ohe.transform(pred_arr[:,0].reshape(-1,1)).toarray()

# print(up_down_label)
#
#
# 训练集和验证集的划分
x_train, x_test, y_train, y_test = \
    train_test_split(pred_arr[:, 1:], up_down_label, test_size=0.3, random_state=12)

# reshape input
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

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

##################################################################################
#普通lstm准确度0.58
###################################################################################
# model = Sequential()
# model.add(LSTM(200, input_shape=(x_train.shape[1], x_train.shape[2])))
# model.add(Dropout(0.3))
# #加入全连接层
# model.add(Dense(3, activation='softmax'))
# #调用compile函数来指定损失函数以及优化器
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# # 开始训练,validation_split=0.2
# history = model.fit(x_train, y_train, validation_split=0.2, epochs=300, batch_size=32)

##################################################################################
#堆叠lstm准确度0.61
###################################################################################
# model = Sequential()
# model.add(LSTM(units=200, input_shape=(x_train.shape[1], x_train.shape[2]),return_sequences=True))
# model.add(LSTM(units=100, return_sequences=False))
# model.add(Dropout(0.3))
# #加入全连接层
# model.add(Dense(3, activation='softmax'))
# #调用compile函数来指定损失函数以及优化器
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# # 开始训练,validation_split=0.2
# history = model.fit(x_train, y_train, validation_split=0.2, epochs=300, batch_size=32)

##################################################################################
#ConvLSTM2D准确度0.57
###################################################################################
# x_train = x_train.reshape(x_train.shape + (1, 1))
# x_test = x_test.reshape(x_test.shape + (1, 1))
#
# model = Sequential()
# model.add(ConvLSTM2D(filters=64, input_shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3],x_train.shape[4]),
#                      kernel_size=(5,5),padding='same'))
#
# # model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
# # model.add(GlobalMaxPooling1D())
# model.add(Flatten())
# model.add(Dropout(0.3))
# #加入全连接层
# model.add(Dense(3, activation='softmax'))
# #调用compile函数来指定损失函数以及优化器
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# # 开始训练,validation_split=0.2
# history = model.fit(x_train, y_train, validation_split=0.2, epochs=300, batch_size=32)

##################################################################################
#cnn*lstm准确度0.62
###################################################################################
# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=3, padding='same', strides=1,
#                  activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
# model.add(MaxPooling1D(pool_size=1))
# model.add(LSTM(200, return_sequences=True))
# model.add(LSTM(100, return_sequences=True))
# model.add(Flatten())
# model.add(Dropout(0.3))
# #加入全连接层
# model.add(Dense(3, activation='softmax'))
# #调用compile函数来指定损失函数以及优化器
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# # 开始训练,validation_split=0.2
# history = model.fit(x_train, y_train, validation_split=0.2, epochs=300, batch_size=32)
##################################################################################
#双向lstm准确度0.59
###################################################################################
#创建一个Sequential模型
model = Sequential()
# #双向LSTM考虑前后
model.add(Bidirectional(LSTM(units=64, return_sequences=False)))#双向LSTM考虑前后词

#加入dropout层避免过拟合
model.add(Dropout(0.3))
#加入全连接层
model.add(Dense(3, activation='softmax'))
#调用compile函数来指定损失函数以及优化器
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 开始训练,validation_split=0.2
history = model.fit(x_train, y_train, validation_split=0.2, epochs=300,batch_size=32)

############################################################################################
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
