
#################### import libraries ####################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#################### import data ####################

data = pd.read_csv("SECTİON3/PREDICTION/odev_tenis.csv")

#################### categoric values ####################

##### encoding

# outlook

from sklearn import preprocessing 

ohe = preprocessing.OneHotEncoder()

outlook = data.iloc[:,0:1].values

outlook_ohe = ohe.fit_transform(outlook).toarray()

# windy-play

windy_play = data.iloc[:,3:].values

windy_play = pd.DataFrame(data=windy_play,index=range(14),columns=["windy","play"])

from sklearn.preprocessing import LabelEncoder

windy_play = windy_play.apply(LabelEncoder().fit_transform)


# one hot encoding

#################### concat ####################

tempature_humidity = data.iloc[:,1:3].values

outlook = pd.DataFrame(data=outlook_ohe,index=range(14),columns=["overcast","rainy","sunny"])

tempature_humidity = pd.DataFrame(data=tempature_humidity,index=range(14),columns=["tempature","humidity"])

# concat 

tempature = tempature_humidity["tempature"]

result = pd.concat([outlook,tempature],axis=1)

result2 = pd.concat([result,windy_play],axis=1)

humidity = tempature_humidity["humidity"]


#################### Train Test Split ####################

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(result2,humidity,test_size=0.33,random_state=0)

########### play kolonu için tahmin 

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


# backward elimination

import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int),values=result2,axis=1)

X_l = result2.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(humidity,X_l).fit()
print(model.summary())

 #  0.593 > p-value
 
X_l = result2.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(humidity,X_l).fit()
print(model.summary())

#  0.387 > p-value

X_l = result2.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(humidity,X_l).fit()
print(model.summary())

#  0.265  > p-value

X_l = result2.iloc[:,[0,1,2]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(play,X_l).fit()
print(model.summary())














