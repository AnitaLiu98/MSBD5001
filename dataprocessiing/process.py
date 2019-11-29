# -*- coding: utf-8 -*-
import pandas as pd
import scipy
from scipy import stats


df=pd.read_csv('../rawdata/train.csv',encoding='utf-8',
               parse_dates=['purchase_date','release_date'])

#########################################################################
#############get_dummies
#########################################################################
for i in range(0,df.shape[0]):
    df.loc[i,'is_free']=int(df.loc[i,'is_free'])

genres = df["genres"].str.get_dummies(",")
genres.columns = genres.columns.map(lambda x: x+'_genres')
print(genres.head(20))
print('genre_num',genres.shape[1])
#categories	tags
categories = df["categories"].str.get_dummies(",")
categories.columns = categories.columns.map(lambda x: x+'_categories')
tags = df["tags"].str.get_dummies(",")
tags.columns = tags.columns.map(lambda x: x+'_tags')
df = pd.concat([df,genres,categories,tags], axis=1, join='outer', join_axes=None, ignore_index=False)
print('len(df.columns):\n',len(df.columns))
#########################################################################
#########################################################################

#检验是否正态分布
df2=df.dropna(how='any')
testres = scipy.stats.kstest(df2['total_negative_reviews'], 'norm')
#缺失值处理，中位数填补
df['total_positive_reviews'] = df['total_positive_reviews'].fillna(df['total_positive_reviews'].median())
df['total_negative_reviews'] = df['total_negative_reviews'].fillna(df['total_negative_reviews'].median())

print(df.loc[5])
#########################################################################
#########################################################################
#时间数据处理 purchase_date	release_date

#游戏购买时间和发行时间的差值：购买时间比发行时间更早：预售时购买
dfd = df['purchase_date'] - df['release_date']
df['date_diff'] = df['purchase_date'] - df['release_date']
print(dfd.head(50))
test2 = dfd
test2= test2.dropna(how='any')
#print(test2.head(9))
test2=pd.to_numeric(test2, errors='coerce')
testres2 = scipy.stats.kstest(test2,'norm')
#print(testres2)
#fillna
dfd = dfd.fillna(dfd.median())
for i in range(0,dfd.shape[0]):
    print(int(dfd.loc[i].days))
    df.loc[i,'date_diff'] = float(dfd.loc[i].days)
#int(dfd.loc[i].days)
print(df['date_diff'].head(50))

#year month
df['y_purchase'] = df['purchase_date'].apply(lambda x: x.year)
df['m_purchase'] = df['purchase_date'].apply(lambda x: x.month)
df['y_release'] = df['release_date'].apply(lambda x: x.year)
df['m_release'] = df['release_date'].apply(lambda x: x.month)
print('newdf',df.head(20))

#某年某月购买的游戏数量
df_ym_pur = df.groupby(['y_purchase','m_purchase']).count().reset_index()[['y_purchase','m_purchase','id']].rename(columns={'id':'pur_num_ym'})
print(df_ym_pur)

#某年购买的游戏数量，
num_y_pur = df.groupby(['y_purchase']).count().reset_index()[['y_purchase','id']].rename(columns={'id':'pur_num_year'})
print(num_y_pur)
#某年购买的游戏平均使用时长
pt_y_pur = df.groupby(['y_purchase']).mean().reset_index()[['y_purchase','playtime_forever']].rename(columns={'playtime_forever':'playtime_purchase_year'})
print(pt_y_pur)
#xx月购买的游戏平均使用总时长
pt_pur_m = df.groupby(['m_purchase']).mean().reset_index()[['m_purchase','playtime_forever']].rename(columns={'playtime_forever':'playtime_purchase_m'})
print(pt_pur_m)
#xx月发行的游戏平均使用总时长
pt_re_m = df.groupby(['m_release']).mean().reset_index()[['m_release','playtime_forever']].rename(columns={'playtime_forever':'playtime_release_m'})
print(pt_re_m)
#########################################################################
#########################################################################


df = pd.merge(df,df_ym_pur, left_on=['y_purchase','m_purchase'], right_on=['y_purchase','m_purchase'], how='left',suffixes=('',''), sort=False)
print(df.columns)

df = pd.merge(df,num_y_pur, left_on='y_purchase', right_on='y_purchase', how='left',suffixes=('',''), sort=False)
print(df.columns)
df = pd.merge(df,pt_pur_m, left_on='m_purchase', right_on='m_purchase', how='left', sort = False)
print(df.columns)
df = pd.merge(df,pt_re_m, left_on='m_release', right_on='m_release', how='left',suffixes=('',''), sort=False)
print(df.columns)

df['pur_num_ym'] = df['pur_num_ym'].fillna(df_ym_pur['pur_num_ym'].median())
df['pur_num_year'] = df['pur_num_year'].fillna(num_y_pur['pur_num_year'].median())
df['playtime_purchase_m'] = df['playtime_purchase_m'].fillna(pt_pur_m['playtime_purchase_m'].median())
print(df.head(6))

'''writer = pd.ExcelWriter('/Users/liuxueling/PycharmProjects/python3pj/5001personal/dataprocessiing/processed data/prefeatures.xlsx',engine='xlsxwriter')
df.to_excel(writer)
writer.save()'''


#drop id.........................
df = df.drop(['id','purchase_date', 'release_date','genres', 'categories','tags','y_purchase','m_purchase'],axis=1)
print(df.head(6))


'''writer = pd.ExcelWriter('/Users/liuxueling/PycharmProjects/python3pj/5001personal/dataprocessiing/processed data/prefeatures_dropold.xlsx',engine='xlsxwriter')
df.to_excel(writer)
writer.save()'''

df.to_csv('../dataprocessiing/processed data/prefeatures_dropold.csv')
#########################################################################
'''#########################################################################
#########################          归一化  #########  ####################
#########################################################################
def regularit(df):
    import pandas as pd
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    for c in columns:
        
        newDataFrame[c] = df[c].apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))
    return newDataFrame

print(df.isnull().any())
newDataFrame = pd.DataFrame(index=df.index)
columns = df.columns.tolist()
for c in columns:
    
    newDataFrame['playtime_forever'] = df['playtime_forever']
    if c!='playtime_forever':
        print(df[c].apply(lambda x: ((x - df[c].min()) / (df[c].max() - df[c].min()))).head(6))
        d = df[c].apply(lambda x: ((x - df[c].min()) / (df[c].max() - df[c].min())))
        # newDataFrame = newDataFrame.drop(c, axis=1)
        newDataFrame[c] = d
#df = regularit(df)
df = newDataFrame
print(newDataFrame.head(6))'''


'''writer = pd.ExcelWriter('/Users/liuxueling/PycharmProjects/python3pj/5001personal/dataprocessiing/processed data/prefeatures_normlized.csv',engine='xlsxwriter')
df.to_excel(writer)
writer.save()'''


#########################################################################################################################
#############            df——test  测试集       ################################################################################
#########################################################################################################################
df_test = pd.read_csv('../rawdata/test.csv',encoding='utf-8',
               parse_dates=['purchase_date','release_date'])

#########################################################################
############# _test get_dummies
#########################################################################
for i in range(0,df_test.shape[0]):
    df_test.loc[i,'is_free']=int(df_test.loc[i,'is_free'])

genres_test = df_test["genres"].str.get_dummies(",")
genres_test.columns = genres_test.columns.map(lambda x: x+'_genres')
print(genres_test.head(20))
print('genre_num_test',genres_test.shape[1])
#categories	tags
categories_test = df_test["categories"].str.get_dummies(",")
categories_test.columns = categories_test.columns.map(lambda x: x+'_categories')
tags_test = df_test["tags"].str.get_dummies(",")
tags_test.columns = tags_test.columns.map(lambda x: x+'_tags')
print(tags_test.columns)
#newcol = list(df_test.columns)+list(genres.columns)+list(categories.columns)+list(tags.columns)
print('len(df_test.columns):\n',len(df_test.columns))
for colname in list(genres.columns)+list(categories.columns)+list(tags.columns):
    if colname in genres_test.columns:
        df_test[colname]= genres_test[colname]
    else :
        df_test[colname] = 0
    if colname in categories_test.columns:
        df[colname] = categories_test[colname]
    else :
        df_test[colname] = 0
    if colname in tags_test.columns:
        df_test[colname]=tags_test[colname]
    else :
        df_test[colname] = 0
print('len(df_test.columns):\n',len(df_test.columns))
print(df_test)

#缺失值处理，中位数填补
df_test['total_positive_reviews'] = df_test['total_positive_reviews'].fillna(df['total_positive_reviews'].median())
df_test['total_negative_reviews'] = df_test['total_negative_reviews'].fillna(df['total_negative_reviews'].median())

#########################################################################
#########################################################################
#时间数据处理 purchase_date	release_date

#游戏购买时间和发行时间的差值：购买时间比发行时间更早：预售时购买
dfd_test = df_test['purchase_date'] - df_test['release_date']
df_test['date_diff'] = df_test['purchase_date'] - df_test['release_date']
print(dfd_test.head(50))

#fillna
dfd_test = dfd_test.fillna(dfd.median())#测试集空值填训练集中位数
for i in range(0,dfd_test.shape[0]):
    df_test.loc[i,'date_diff'] = float(dfd_test.loc[i].days)
#int(dfd.loc[i].days)
print(df_test['date_diff'])


#year month
df_test['y_purchase'] = df_test['purchase_date'].apply(lambda x: x.year)
df_test['m_purchase'] = df_test['purchase_date'].apply(lambda x: x.month)
df_test['y_release'] = df_test['release_date'].apply(lambda x: x.year)
df_test['m_release'] = df_test['release_date'].apply(lambda x: x.month)

#########################################################################
#########################################################################


df_test = pd.merge(df_test,df_ym_pur, left_on=['y_purchase','m_purchase'], right_on=['y_purchase','m_purchase'], how='left',suffixes=('',''), sort=False)
print(df_test.columns)

df_test = pd.merge(df_test,num_y_pur, left_on='y_purchase', right_on='y_purchase', how='left',suffixes=('',''), sort=False)
print(df_test.columns)
df_test = pd.merge(df_test,pt_pur_m, left_on='m_purchase', right_on='m_purchase', how='left', sort = False)
print(df_test.columns)
df_test = pd.merge(df_test,pt_re_m, left_on='m_release', right_on='m_release', how='left',suffixes=('',''), sort=False)
print(df_test.columns)

df_test['pur_num_ym'] = df_test['pur_num_ym'].fillna(df_ym_pur['pur_num_ym'].median())
df_test['pur_num_year'] = df_test['pur_num_year'].fillna(num_y_pur['pur_num_year'].median())
df_test['playtime_purchase_m'] = df_test['playtime_purchase_m'].fillna(pt_pur_m['playtime_purchase_m'].median())
print(df_test[['pur_num_ym','pur_num_year','playtime_purchase_m']])

#drop id.........................
df_test = df_test.drop(['id','purchase_date', 'release_date','genres', 'categories','tags','y_purchase','m_purchase'],axis=1)
print(df_test)

df=df.drop('playtime_forever',axis=1)
print(len(list(df_test.columns)))
print(len(list(df.columns)))
print(list(df_test.columns)==list(df.columns))
for i in list(df.columns):
    if i not in list(df_test.columns):
        print(i)

print(list(df_test.columns))
print(list(df.columns))



#writer = pd.ExcelWriter('/Users/liuxueling/PycharmProjects/python3pj/5001personal/dataprocessiing/processed data/test_feature.xlsx',engine='xlsxwriter')
df_test.to_csv('../dataprocessiing/processed data/test_feature.csv')
