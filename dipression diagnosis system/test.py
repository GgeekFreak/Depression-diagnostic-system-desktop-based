# age , blood pressure , urine specific gravity , albumine , serum creatine urine level

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('kidney_disease.csv',header=None,skiprows=1)




#
# xtest = [65,60,1.02,0,0]
# xtest = np.array(xtest).reshape(1,-1)
# print(xtest)
#
# df = pd.DataFrame(data)
# df2 = df.dropna(axis=0,how='any')
#
# X = df2.iloc[:,1:6]
# Y = df2.iloc[:,-1]
# clf = svm.SVC(kernel='linear')
#
# x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)
# clf.fit(x_train,y_train)
# print(x_test)
# preds = clf.predict(xtest)
#
#
#
# print(preds)

