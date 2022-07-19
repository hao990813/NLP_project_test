import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split # 进行训练和测试样本的分割(sklearn 0.23.1)
from sklearn.preprocessing import OneHotEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
                             'current_low','current_vol','公告日期','代码']]

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
                             'current_low','current_vol','date','code']]
# print(df_pred.head())

#归一化操作
transfer = MinMaxScaler()
pred_arr = transfer.fit_transform(df_pred)
# print(pred_arr)

# #将标签进行one-hot编码
# up_down_pred= np.unique(pred_arr[:,0])
# ohe = OneHotEncoder()
# ohe.fit([[up_down_pred[0]],[up_down_pred[1]],[up_down_pred[2]]])
# up_down_label = ohe.transform(pred_arr[:,0].reshape(-1,1)).toarray()

##################################################################################
#使用LDA线性判别准确度0.301
###################################################################################
# #将小数转换为整数
# label_output = pred_arr[:, 0]*10
# # 训练集和验证集的划分
# x_train, x_test, y_train, y_test = \
#     train_test_split(pred_arr[:, 1:], label_output, test_size=0.3, random_state=12)
#
# model = LinearDiscriminantAnalysis()
# model.fit(x_train, y_train)
# # 预测测试数据集，得出准确率
# y_predict = model.predict(x_test)
# print("预测测试集类别：", y_predict)
# print("准确率为：", model.score(x_test, y_test))

# ##################################################################################
# #使用knn准确度0.502
# ###################################################################################
# # 将小数转换为整数
# label_output = pred_arr[:, 0]*10
# # 训练集和验证集的划分
# x_train, x_test, y_train, y_test = \
#     train_test_split(pred_arr[:, 1:], label_output, test_size=0.3, random_state=12)
#
# model = KNeighborsClassifier()
# #设置超参数
# param = {"n_neighbors": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
# #使用GridSearchCV进行参数的调优
# gc = GridSearchCV(model, param_grid=param, cv=2)
# model.fit(x_train, y_train)
# # 预测测试数据集，得出准确率
# y_predict = model.predict(x_test)
# print("预测测试集类别：", y_predict)
# print("准确率为：", model.score(x_test, y_test))

##################################################################################
#使用SVM准确度0.495
###################################################################################
# # 将小数转换为整数
# label_output = pred_arr[:, 0]*10
# # 训练集和验证集的划分
# x_train, x_test, y_train, y_test = \
#     train_test_split(pred_arr[:, 1:], label_output, test_size=0.3, random_state=12)
#
# model = SVC()
# model.fit(x_train, y_train)
# # 预测测试数据集，得出准确率
# y_predict = model.predict(x_test)
# print("预测测试集类别：", y_predict)
# print("准确率为：", model.score(x_test, y_test))

##################################################################################
#使用决策树准确度0.435
###################################################################################
# # 将小数转换为整数
# label_output = pred_arr[:, 0]*10
# # 训练集和验证集的划分
# x_train, x_test, y_train, y_test = \
#     train_test_split(pred_arr[:, 1:], label_output, test_size=0.3, random_state=12)
#
# model = DecisionTreeClassifier()
# #设置超参数
# param = {"max_depth": [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]}
# #使用GridSearchCV进行参数的调优
# gc = GridSearchCV(model, param_grid=param, cv=2)
# model.fit(x_train, y_train)
# # 预测测试数据集，得出准确率
# y_predict = model.predict(x_test)
# print("预测测试集类别：", y_predict)
# print("准确率为：", model.score(x_test, y_test))

# ##################################################################################
# #使用随机森林准确度0.508
# ###################################################################################
# 将小数转换为整数
label_output = pred_arr[:, 0]*10
# 训练集和验证集的划分
x_train, x_test, y_train, y_test = \
    train_test_split(pred_arr[:, 1:], label_output, test_size=0.3, random_state=12)

model = RandomForestClassifier()
#设置超参数
param = {"n_estimators": [100,200,300,400,500,600,700,800,900,1000,1100,1200],
         "max_depth": [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]}
#使用GridSearchCV进行参数的调优
gc = GridSearchCV(model, param_grid=param, cv=2)
model.fit(x_train, y_train)
# 预测测试数据集，得出准确率
y_predict = model.predict(x_test)
print("预测测试集类别：", y_predict)
print("准确率为：", model.score(x_test, y_test))




