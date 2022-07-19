from data_with_sentiment import *

#######################################################################
                #与数据库进行连接，筛选出来对应的股票收盘价数据
######################################################################
#提取出去重的股票的代码list 方便在数据库中进行查询
stock_drop_dup = df_news_comine['代码'].drop_duplicates().tolist()

#调整新闻的时间，因为我们需要的是这个新闻日期的后一天收盘价数据，并且要考虑到是否为交易日的因素
#选取要筛选的股票时间
#首先筛选出来去重后的时间，数据形式为datetime
news_time = df_news_comine['公告日期'].drop_duplicates().tolist()

#将datetime格式的数据转换为int类型，作为参数传入到数据库中
news_input_time = []
for i in range(len(news_time)):
      news_input_time.append(int(news_time[i] .strftime('%Y%m%d')))

#将得出来的stock_check_time进行排序
news_input_time = sorted(news_input_time)
# print(stock_check_time)

#找到新闻的最后一天，拿到股票中去进行匹配新闻最后一天的下一个交易日
#建立与股票数据库的连接
connection_1 = sqlalchemy.create_engine("mysql+pymysql://guest:MH@123456@172.31.50.57:3306/astocks")
#sql语句只输出后一个交易日的日期
sql_1 = "SELECT td FROM market " \
      "WHERE (td > {}) and  (codenum = '000001.SZ') limit 1".format(news_input_time[-1])

#read_sql直接转换成df形式
df_temp_time_1 = pd.read_sql(sql_1, connection_1)
#找到最后一个交易时间并进行类型转换
last_trading_day = df_temp_time_1['td'][0]
#将最后一个交易日加入到时间list中
news_input_time.append(last_trading_day)
# print(stock_check_time)
# print(last_trading_day)

#找到新闻的第一天，进行判断，若当天为交易日则设定当天值，不是则往前推一天
#建立与股票数据库的连接
connection_2 = sqlalchemy.create_engine("mysql+pymysql://guest:MH@123456@172.31.50.57:3306/astocks")
#sql语句只输出后一个交易日的日期
sql_2 = "SELECT td FROM market " \
      "WHERE (td <= {}) and  (codenum = '000001.SZ') limit 1".format(news_input_time[0])

#read_sql直接转换成df形式
df_temp_time_2 = pd.read_sql(sql_2, connection_2)
#找到第一一个交易时间并进行类型转换
first_trading_day = df_temp_time_2['td'][0]
#判断此日期是否在list中，不在的话则加入
if first_trading_day not in news_input_time:
    news_input_time.insert(0,first_trading_day)
# print(stock_check_time)
# print(last_trading_day)

#重新建立与股票数据库的连接
connection = sqlalchemy.create_engine("mysql+pymysql://guest:MH@123456@172.31.50.57:3306/astocks")
#设置sql语句，将出现的股票当作元组传入，时间也作为元组在数据库里直接进行筛选
sql = "SELECT td,codenum,close,open,high,low,vol FROM market " \
      "WHERE td in {} and codenum in {}".format(tuple(news_input_time), tuple(stock_drop_dup))

#read_sql直接转换成df形式
df_stock_data = pd.read_sql(sql, connection)

#将td时间从int格式转换为datetime格式
df_stock_data['td'] = df_stock_data['td'].astype('str')
df_stock_data['td'] = pd.to_datetime(df_stock_data['td'],infer_datetime_format=True)

#筛选出来去重后的股票数据时间，数据形式为datetime，方便与之前的新闻时间进行匹配
trading_day = df_stock_data['td'].drop_duplicates().tolist()

# print(news_time)
# print(trading_day)
#给df_news_comine增加一列current_trading_day,和一列next_trading_day
list_current_trading_day = []
list_next_trading_day = []

#循环来进行current_trading_day和next_trading_day的匹配
for i in range(len(df_news_comine)):
    for j in range(len(trading_day)-1):
        if df_news_comine['公告日期'][i] == trading_day[j]:
            list_current_trading_day.append(trading_day[j])
            list_next_trading_day.append(trading_day[j+1])
            continue
        else:
            if (df_news_comine['公告日期'][i] > trading_day[j]) and (df_news_comine['公告日期'][i] < trading_day[j+1]):
                list_current_trading_day.append(trading_day[j])
                list_next_trading_day.append(trading_day[j+1])
                continue

#将current_trading_day与list_next_trading_day添加到dataframe中去
df_news_comine['current_trading_day'] = list_current_trading_day
df_news_comine['next_trading_day'] = list_next_trading_day
# print(df_news_comine)

#分别将df_news_comine与df_stock_data进行两次left_join来找到所以对应的current和next的trading_day的close值
#先于current_trading_day进行merge
df_temp_full_data = pd.merge(df_news_comine,df_stock_data,left_on=['代码','current_trading_day'],right_on=['codenum','td'],how='left')
#删除因为merge所产生的重复的列
df_temp_full_data = df_temp_full_data.drop(columns=['codenum','td'])
# #将columns重新命名
# df_temp_full_data.columns = ['公告日期','代码','公告标题','current_trading_day','next_trading_day','current_close']
# # print(df_full_data)
#再和next_trading_day进行merge
df_full_data = pd.merge(df_temp_full_data,df_stock_data,left_on=['代码','next_trading_day'],right_on=['codenum','td'],how='left')
#删除因为merge所产生的重复的列
df_full_data = df_full_data.drop(columns=['codenum','td'])
#将columns重新命名
df_full_data.columns = ['公告日期','代码','公告标题','pos','neg','current_trading_day',
                        'next_trading_day','current_close','current_open','current_high',
                        'current_low','current_vol','next_close','next_open','next_high',
                        'next_low','next_vol']
#删除有nan值所在的行
df_full_data = df_full_data.dropna()
#重新索引
df_full_data = df_full_data.reset_index(drop=True)
# print(df_full_data)

# #进行判断涨跌情况
#使用np.where嵌套看是涨还是跌，涨赋值1跌赋值-1，平赋值0
df_full_data['up_down'] = (df_full_data['next_close'] - df_full_data['current_close'])\
                          /df_full_data['current_close']
quantile_top = df_full_data['up_down'].quantile(0.33)
quantile_down = df_full_data['up_down'].quantile(0.66)
df_full_data['up_or_down'] = np.where(df_full_data['up_down'] > quantile_down, 1, np.where(df_full_data['up_down'] < quantile_top,-1,0))
# #进行情感判断
# df_full_data['pos_neg'] = df_full_data['pos'] - df_full_data['neg']
#删除在pos与neg均为0的
indexNames=df_full_data[(df_full_data['pos']==0)&(df_full_data['neg']==0)].index
df_full_data.drop(indexNames,inplace=True)
df_full_data = df_full_data.reset_index(drop=True)
df_full_data['pos_neg'] = df_full_data['pos'] - df_full_data['neg']
# #将公告标题list转为string
# df_full_data['公告标题'] = ''.join(df_full_data['公告标题'].tolist())

#将同一天同一股票的新闻拼接成str类型的一句话
temp_store = []
for i in range(len(df_full_data['公告标题'])):
    for j in range(len(df_full_data['公告标题'][i])):
        for z in range(len(df_full_data['公告标题'][i][j])):
            #转化为str类型
            temp_store.append(str(df_full_data['公告标题'][i][j][z]))
    df_full_data._set_value(i, '公告标题', ''.join(temp_store))
    temp_store = []

#进行情感的划分，依据pos的次数是否大于neg的词数
df_full_data['pos_or_neg'] = np.where(df_full_data['pos_neg'] > 0, 1,
                                      np.where(df_full_data['pos_neg'] < 0,-1,0))

#

# print(df_full_data.head())
# print(type(df_full_data['公告标题'][0][0]))

# #删除没用的列
df_full_data = df_full_data.drop(columns=['pos','neg','current_trading_day','next_trading_day','next_open','next_high','next_low','next_vol','up_down','pos_neg'])
# print(len(df_full_data))
# print(df_full_data['pos_or_neg'][df_full_data['pos_or_neg']== -1].count())
# print(df_full_data['pos_or_neg'][df_full_data['pos_or_neg']== 0].count())
# print(df_full_data['pos_or_neg'][df_full_data['pos_or_neg']== 1].count())

# 以pickle的格式保存dataframe，原格式才能被保存
df_full_data.to_csv('df_full_data.csv',encoding='utf_8_sig')

