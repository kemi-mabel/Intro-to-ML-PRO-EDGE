# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 07:49:13 2019

@author: oluwakemmix
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:26:14 2019

@author: oluwakemmix
"""
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing our data set
dataset=pd.read_csv(r'C:\Users\oluwakemmix\Desktop\ProEdge\companiesForML.csv')

#view your dataset
dataset.tail()
dataset.head()
#handling missing data

#splitting into dependent and independent varaible
print(x2)
x1=dataset.iloc[:,0].values 
x2=dataset.iloc[:,1].values 
x3=dataset.iloc[:,2].values
y=dataset.iloc[:,4].values

#splitting our data into testing and training set
from sklearn.model_selection import train_test_split
x1_train,x1_test,y_train,y_test=train_test_split(x1,y,test_size=0.3,random_state=0)
x2_train,x2_test,y_train,y_test=train_test_split(x2,y,test_size=0.3,random_state=0)
x3_train,x3_test,y_train,y_test=train_test_split(x3,y,test_size=0.3,random_state=0)

x1_train = x1_train.reshape(-1,1)
x2_train = x2_train.reshape(-1,1)
x3_train = x3_train.reshape(-1,1)

#importing the linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x1_train,y_train)
regressor.fit(x2_train,y_train)
regressor.fit(x3_train,y_train)

 
#predicting your model
y1_pred=regressor.predict(x1_train)
y1_pred
y2_pred=regressor.predict(x2_train)
y2_pred
y3_pred=regressor.predict(x3_train)
y3_pred


#how to visualize your model
plt.scatter(x1_train,y_train,color='purple')
plt.scatter(x2_train,y_train,color='yellow')
plt.scatter(x3_train,y_train,color='red')

# multiple line plot
plt.plot(x1_train,y1_pred,color='purple')
plt.scatter(x3_train,y_train,color='red')

plt.scatter(x2_train,y_train,color='yellow')
plt.plot(x2_train,y2_pred,color='yellow')

plt.plot(x3_train,y3_pred,color='red')
plt.scatter(x3_train,y_train,color='red')

plt.title('Profit vs all(Training set)')
plt.x1label('R&D spencer')
plt.x2label('admin')
plt.x3label('marketing')
plt.ylabel('profit')
plt.show()

