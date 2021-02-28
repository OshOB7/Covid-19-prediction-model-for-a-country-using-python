# Covid-19-prediction-model-for-a-country-using-python
# importing required libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import seaborn as sns
# Disable warnings
import warnings
warnings.filterwarnings('ignore')

pip install numpy pandas sklearn 

from sklearn.preprocessing import PolynomialFeatures

#Read the data
df=pd.read_csv('C:\\Users\PB\Desktop\EXL EQ\\EQ_2021_Data_Sample.csv')

#Get shape and head
df.shape
df.head()

df.shape

print (df.columns)

dates = df['date']
date_format = [pd.to_datetime(d) for d in dates]

data=df[['countyFIPS', 'stateFIPS', 'date', 'confirmed_cases','C_TOT_POP','new_test_rate', 'new_test_count']]

print('_'*20);print('HEAD');print('_'*20)
print(data.head())

print('_'*20);print('PREPARE DATA');print('_'*20)
x=np.array(data['countyFIPS']).reshape(-1,1)
y=np.array(data['confirmed_cases']).reshape(-1,1)
plt.plot(y,'-m')
plt.show()
polyFeat= PolynomialFeatures(degree=2)
x=polyFeat.fit_transform(x)
print(x)

polyFeat= PolynomialFeatures(degree=) 
x=polyFeat.fit_transform(x) 
print(x)

print('-'*20);print('TRAINING DATA');print('-'*20)
model=linear_model.LinearRegression()
model.fit(x,y)
accuracy=model.score(x,y)
print(f'Accuracy:{(accuracy)}')

