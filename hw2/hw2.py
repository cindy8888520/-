import pandas as pd
import numpy as np
#%cd C:\\Users\\Yan Ping Wu\\Desktop\\hw2
# 讀取資料
data = pd.read_excel('106年新竹站_20180309.xls')

'''資料前處理 '''
#  a. 取出10.11.12月資料
data = data.loc[data['日期'] >= '2017/10/01'].reset_index(drop=True)
# 刪除測站欄位
data.drop(columns='測站', inplace=True)


#  c. NR表示無降雨，以0取代
for i, row in data.iterrows():
    for col in data.columns:
        if data.loc[i, col] == 'NR':
            data.loc[i, col] = 0


#  b. 缺失值以及無效值以前後一小時平均值取代 (如果前一小時仍有空值，再取更前一小時)

# 將包含#、*、x的缺失值，找出來並補上nan
data.iloc[:, 2:26] = data.iloc[:, 2:26].replace(
    r'[#*x]', np.nan, regex=True)
    
    
    

# 填補缺失值
data_new=data.pivot('測項','日期').copy()    
data_new.columns.rename(['小時','日期'],inplace=True)    
data_new.sort_index(axis=1,level=['日期','小時'],ascending=True,inplace=True)   

df_hrs = pd.concat([data_new.ffill(axis=1),data_new.bfill(axis=1)]).groupby(level=0).mean()  

df_hrs=df_hrs.stack(level=1).sort_index(axis=0,level=['日期','測項'])

data=df_hrs.reset_index().copy()
    

# Check for missing values
data.isnull().any()


#  d. 將資料切割成訓練集(10.11月)以及測試集(12月)

train_data = data.loc[data['日期'] < '2017/12/01'].reset_index(drop=True)
test_data = data.loc[data['日期'] >= '2017/12/01'].reset_index(drop=True)

#  e. 製作時序資料: 將資料形式轉換為行(row)代表18種屬性，欄(column)代表逐時數據資料 **hint: 將訓練集每18行合併，轉換成維度為(18,61*24)的DataFrame(每個屬性都有61天*24小時共1464筆資料)
train_data = train_data.pivot(index='測項', columns='日期')
test_data = test_data.pivot(index='測項', columns='日期')
''' 時間序列 '''
#  a. 取6小時為一單位切割，例如第一筆資料為第0~5小時的資料(X[0])，去預測第6小時的PM2.5值(Y[0])，下一筆資料為第1~6小時的資料(X[1])去預測第7小時的PM2.5值(Y[1])
#  b. X請分別取
# 	1. 只有PM2.5
train_data.columns.rename(['日期', '小時'], level=[1, 0], inplace=True)

train_data_lm = train_data.loc[train_data.index == 'PM2.5', :]
# Create design matrix
for i in range(1, 7):
    ts = train_data.loc[train_data.index ==
                        'PM2.5', :].shift(i, axis='columns')
    train_data_lm = pd.concat([train_data_lm, ts], axis=0)
# reset index and drop NaN values
train_data_lm = train_data_lm.reset_index(drop=True)
train_data_lm.dropna('columns', inplace=True)

# 	2. 所有18種屬性
train_data_all=train_data.T
for inx in train_data.index:
    for i in range(1,7):
        train_data_all=pd.concat([train_data_all,train_data.loc[inx,:].shift(i).rename(inx+'_LAG'+str(i))],axis=1)

# drop nan values
train_data_all.dropna('rows',inplace=True)

#  c. 使用兩種模型 Linear Regression 和 Random Forest Regression 建模

""" 只有PM2.5 """
# linear regression
import numpy as np
from sklearn.linear_model import LinearRegression
train_data_lm = train_data_lm.T
train_data_lm.rename(columns=lambda i: 'LAG'+str(i), inplace=True)
lm_Y = train_data_lm.loc[:, 'LAG0']
lm_X = train_data_lm.drop(columns='LAG0')
reg = LinearRegression().fit(lm_X, lm_Y)
reg.coef_
reg.intercept_
# Linear Regression Prediction

test_data_lm = test_data.loc[test_data.index == 'PM2.5', :]
# Create design matrix
for i in range(1, 7):
    ts = test_data.loc[test_data.index == 'PM2.5', :].shift(i, axis='columns')
    test_data_lm = pd.concat([test_data_lm, ts], axis=0)
# reset index and drop NaN values
test_data_lm = test_data_lm.reset_index(drop=True)
test_data_lm.dropna('columns', inplace=True)
test_data_lm = test_data_lm.T
test_data_lm.rename(columns=lambda i: 'LAG'+str(i), inplace=True)
pred_lm = reg.predict(test_data_lm.drop(columns='LAG0'))


# Random Forest (PM2.5)

from sklearn.ensemble import RandomForestRegressor
rf_PM = RandomForestRegressor(n_estimators=500,max_leaf_nodes=5,max_features=2,random_state=0)
rf_PM.fit(lm_X,lm_Y)
pred_rf=rf_PM.predict(test_data_lm.drop(columns='LAG0'))

""" 所有18種屬性 """
# Linear Regression using 18*6 features

from sklearn.linear_model import LinearRegression
train_data_all_X=train_data_all.drop(columns=train_data.index)
train_data_all_Y=train_data_all.loc[:,'PM2.5']
reg1=LinearRegression().fit(train_data_all_X,train_data_all_Y)
reg1.coef_
reg1.intercept_

# Create testing data
test_data_all=test_data.T
for inx in test_data.index:
     for i in range(1,7):
          test_data_all=pd.concat([test_data_all,test_data.loc[inx,:].shift(i).rename(inx+'_LAG'+str(i))],axis=1)
test_data_all.dropna('rows',inplace=True)          
test_data_all_X=test_data_all.drop(columns=train_data.index)
# Prediction using 18*6 predictors
from sklearn.metrics import mean_absolute_error
pred_all_lm=reg1.predict(test_data_all_X)

# Random Forest using 18*6 features
from sklearn.ensemble import RandomForestRegressor
rf_all = RandomForestRegressor(n_estimators=500,max_leaf_nodes=5,max_features=2,random_state=0)
rf_all.fit(train_data_all_X,train_data_all_Y)
pred_all_rf=rf_all.predict(test_data_all_X)


#  d. 用測試集資料計算MAE (會有4個結果，2種模型*2種X資料)
# metrics MAE
from sklearn.metrics import mean_absolute_error
# (1)只有PM2.5 的Time Series Regression
print(mean_absolute_error(test_data_lm['LAG0'], pred_lm))

# (2)只有PM2.5的Random Forest
print(mean_absolute_error(test_data_lm['LAG0'], pred_rf))
# (3)所有18個屬性的Time Series Regression

print(mean_absolute_error(test_data_all['PM2.5'], pred_all_lm))
# (4)所有18個屬性的Random Forest
print(mean_absolute_error(test_data_all['PM2.5'],pred_all_rf))


