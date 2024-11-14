import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df=pd.read_csv('logistic_regression_data.csv')
x_train,x_test,y_train,y_test=train_test_split(df[['feature']],df.label,test_size=0.1)
#plt.scatter(df.feature,df.label)
#plt.show()

model=LogisticRegression()
model.fit(x_train,y_train)

#print(x_test)
#print(model.predict([[50]]))

plt.scatter(df.feature,df.label)
plt.plot(x_train,y_train)
plt.show()