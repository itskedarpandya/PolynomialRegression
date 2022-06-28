# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 21:33:18 2022

@author: Kedar Pandya
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
#loading the data
data = pd.read_csv('Position_Salaries.csv')
print(data)

#%%
#seperating the data in dependent and independent variables
X = data.iloc[:,1:-1]
y = data.iloc[:,-1]
print('The dependent variables are:\n',y)
print('The independent variables are:\n',X)

#%%
#as there is not much data in the data set we will not split the data set and apply the models
#directly to the dataset

#first we apply the Simple Linear regression as we will later compare it with Poly regression
from sklearn.linear_model import LinearRegression
sreg = LinearRegression()
sreg.fit(X,y)
pred = sreg.predict(X)

    
#%%
#first creating the polyomial variables
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

#creating a new linear regression model
preg = LinearRegression()
preg.fit(X_poly,y)

#%%
#visualzing the linear regression result

plt.scatter(X,y,color = 'red')
plt.plot(np.array(X),pred, color = 'blue')
plt.title('Truth or Bluff (Linear)')
plt.xlabel('Position level')
plt.ylabel('Salaries')
plt.show()

#%%
#visualizing the polynomial regression result

plt.scatter(X,y,color = 'red')
plt.plot(np.array(X),preg.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial X^4)')
plt.xlabel('Position level')
plt.ylabel('Salaries')
plt.show()
#%%
#value for the 6.5 input will be
print(sreg.predict([[6.5]])) #we need a 2d array that is why we but [[]]
print(preg.predict(poly_reg.fit_transform([[6.5]]))) 
















