# -*- coding: utf-8 -*-

# 1 read data
import pandas as pd
from pandas import DataFrame
data=pd.read_csv('character-deaths.csv',sep=',')

# 2-1 取代空值
data.fillna(0,inplace=True)

# 2-2 Book of Death 轉成1
data['Book of Death']=[1 if d>0 else 0 for d in data['Book of Death']]


# 2-3 Allengiances 產生Dummy

data=pd.concat([data,pd.get_dummies(data['Allegiances'])],axis=1)
# 2-4 Split data (75% vs 25%)

# delete unused variables
data.drop(columns=['Allegiances','Name','Death Year','Death Chapter'],inplace=True)


import sklearn
from sklearn.model_selection import train_test_split

data_train,data_test=train_test_split(data)

# 3 Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(max_features='sqrt')
tree.fit(X=data_train.drop(columns=['Book of Death']),y=data_train['Book of Death'])

# prediction
pred_vals=tree.predict(data_test.drop(columns=['Book of Death']))

# 4 Confusion Matrix

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print("Confusion Matrix:\n")
confusion_matrix(data_test['Book of Death'],pred_vals,[1,0])

# accuracy,percision,recall for 1 (death,not survived)
print("accuracy score is:\n")
accuracy_score(data_test['Book of Death'],pred_vals)
print("precision score is:\n")
precision_score(data_test['Book of Death'],pred_vals)
print("recall score is:\n")
recall_score(data_test['Book of Death'],pred_vals)

# for different class (0 and 1) compute precision recall fscore
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(data_test['Book of Death'],pred_vals)

# Plot

from sklearn.tree import export_graphviz

export_graphviz(tree,out_file='tree.dot',filled=True,leaves_parallel=True,rounded=True,feature_names=data_train.drop(columns=['Book of Death']).columns,max_depth=5)

