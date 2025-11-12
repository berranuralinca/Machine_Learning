
#################### import libraries ####################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#################### import data ####################

data = pd.read_csv("SECTİON3/PREDICTION/data.csv")

#################### data preprocessing ####################

height_weight = data[['boy','kilo']]
print(height_weight)

#################### missing values ####################

#################### categoric values ####################

##### encoding

# label encoding

sex = data.iloc[:,-1].values

from sklearn import preprocessing 

le = preprocessing.LabelEncoder()

sex_le = le.fit_transform(sex)

# one hot encoding

from sklearn import preprocessing 

ohe = preprocessing.OneHotEncoder()

country = data.iloc[:,0:1].values

country_ohe = ohe.fit_transform(country).toarray()

#################### concat ####################

result_country = pd.DataFrame(data=country_ohe,index=range(22),columns=["fr","tr","us"])

height_weight = data.iloc[:,1:3].values

result_height_weight = pd.DataFrame(data=height_weight,index=range(22),columns=["height","weight"])

age = data.iloc[:,3].values

result_age = pd.DataFrame(data=age,index=range(22),columns=["age"])

result_sex = pd.DataFrame(data=sex_le,index=range(22),columns=["sex"])

s = pd.concat([result_country,result_height_weight],axis=1)

s2 = pd.concat([s,result_age],axis=1)

s3 = pd.concat([s2,result_sex],axis=1)

#################### Train Test Split ####################

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s2,result_sex,test_size=0.33,random_state=0)

###### cinsiyet kolonu için tahmin

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#####  boy kolonu için tahmin
 
boy = s3.iloc[:,3:4].values

boy = pd.DataFrame(data=boy,index=range(22),columns=["boy"])

left = s3.iloc[:,:3].values

left = pd.DataFrame(data=left,index=range(22),columns=["fr","tr","us"])

right = s3.iloc[:,4:].values

right = pd.DataFrame(data=right,index=range(22),columns=["kilo","yas","cinsiyet"])

data2 = pd.concat([left,right],axis=1)

#################### Train Test Split ####################

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(data2,boy,test_size=0.33,random_state=0)

###### boy kolonu için tahmin

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


######### backward elimination #########

import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int),values=data2,axis=1)

X_l = data2.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())

##  0.717 > p-value

X_l = data2.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())













