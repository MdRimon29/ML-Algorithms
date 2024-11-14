import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris=load_iris()

#print(dir(iris))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3)

#print(len(x_train))

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(x_train,y_train)

ms=model.score(x_test,y_test)
print(ms)
mp=model.predict(x_test)

#print(mp)