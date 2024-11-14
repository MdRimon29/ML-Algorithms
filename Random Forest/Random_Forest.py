import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

####load data
iris=load_iris()    
#print(dir(iris))
df=pd.DataFrame(iris.data,columns=iris.feature_names)
#print(df.head())
df['target']=iris.target
#print(df.head())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df.drop(['target'],axis='columns'),iris.target,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=40)
model.fit(x_train,y_train)

ms=model.score(x_test,y_test)
print(ms)
