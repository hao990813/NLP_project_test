from data_merge import *

#删除空值
df_full_data.dropna(inplace=True)

#KeyedVectors实现实体和向量之间的映射,使用gensim加载预训练中文分词embedding
cn_model = KeyedVectors.load_word2vec_format('./sgns.zhihu.bigram', binary=False)
#现在可以将文本转换成对应的索引
for i in range(len(df_full_data['公告标题'])):
    df_full_data._set_value(i, '公告标题',
                            [k for k in jieba.cut(re.sub('[^\u4e00-\u9fa5]+|公告', '', df_full_data['公告标题'][i])) if
                             k not in stopword_list])
    for j,k in enumerate(df_full_data['公告标题'][i]):
        try:
            # 将词转换为索引index
            df_full_data['公告标题'][i][j] = cn_model.vocab[k].index
        except KeyError:
            # 如果词不在字典中，则输出0
            df_full_data['公告标题'][i][j] = 0
# print(df_full_data.head(30))

#每段标题的长度是不一样的，我们对长度进行标准化,先初始化一个list记录长度
temp_len = []
for i in range(len(df_full_data['公告标题'])):
        temp_len.append(len(df_full_data['公告标题'][i]))

#设置最大的标题长度，并进行标题长度的统一化
# 取tokens平均值并加上两个tokens的标准差，
# 假设tokens长度的分布为正态分布，则25这个值可以涵盖94%左右的样本
max_tokens = np.mean(np.array(temp_len)) + 2 * np.std(np.array(temp_len))
# max_tokens = np.mean(np.array(temp_len))+ np.std(np.array(temp_len))
max_tokens = int(np.max(max_tokens))
# print(max_tokens)
# print(np.sum(np.array(temp_len) < max_tokens) / len(np.array(temp_len)))
# print(np.max(np.array(temp_len)))
# print(np.mean(np.array(temp_len)))

#使用pad_sequences来进行填充和剪枝
train_pad = pad_sequences(df_full_data['公告标题'].tolist(), maxlen=max_tokens,padding='post', truncating='post')
# df_full_data['公告标题'] = list(train_pad)

#对标签进行one-hot编码，转换成神经网络能够接受的格式
ohe = OneHotEncoder()
ohe.fit([[-1],[0],[1]])
senti_label = ohe.transform(df_full_data['pos_or_neg'].values.reshape(-1,1)).toarray()
#划分测试集与训练集
x_train, x_test, y_train, y_test = train_test_split(train_pad,senti_label,test_size=0.3,random_state=12)

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

# 现在我们来为模型准备embedding matrix（词向量矩阵），
# 根据keras的要求，我们需要准备一个维度为(numwords, embeddingdim)的矩阵num words代表我们使用的词汇的数量，
# emdedding dimension在我们现在使用的预训练词向量模型中是300，
# 每一个词汇都用一个长度为300的向量表示
num_words = len(cn_model.vocab)
# 初始化embedding_matrix，之后在keras上进行应用
embedding_matrix = np.zeros((num_words, 300))
# embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
# 维度为 len(cn_model.vocab) * 300
for i in range(num_words):
    embedding_matrix[i,:] = cn_model[cn_model.index2word[i]]
embedding_matrix = embedding_matrix.astype('float32')

#创建一个Sequential模型
model = Sequential()
# 模型第一层为embedding，trainable=False因为embedding_matrix下载后已经训练好了
model.add(Embedding(num_words,300,weights=[embedding_matrix],input_length=max_tokens,trainable=False))
#双向LSTM考虑前后
model.add(Bidirectional(LSTM(units=64, return_sequences=False)))#双向LSTM考虑前后词
#加入dropout层避免过拟合
model.add(Dropout(0.5))
#加入全连接层
model.add(Dense(3, activation='softmax'))
#调用compile函数来指定损失函数以及优化器
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 开始训练,validation_split=0.2
history = model.fit(x_train, y_train, validation_split=0.2, epochs=50,batch_size=32)
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

# #保存模型
# model.save('./word2vec_text_classifier')
# #加载模型
# model = tf.keras.models.load_model('./word2vec_text_classifier')