# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing our data set
dataset=pd.read_csv(r'C:\Users\oluwakemmix\Desktop\ProEdge\Salary_Data.csv')

#view your dataset
dataset.tail()
dataset.head()

#splitting into dependent and independent varaible
x=dataset.iloc[:,-1].values
y=dataset.iloc[:,1].values

print(x)
#handling missing data
dataset.fillna(dataset['Annual Salary(Naira)']).mean()
dataset

#splitting our data into testing and training set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
#importing the linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


#predicting your model
y_pred=regressor.predict(x_train)
y_pred

#how to visualize your model
plt.scatter(x_train,y_train,color='purple')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()





