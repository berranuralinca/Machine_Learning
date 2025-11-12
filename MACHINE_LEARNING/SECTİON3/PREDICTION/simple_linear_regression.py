
#################### import libraries ####################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#################### import data ####################

data = pd.read_csv("MACHINE_LEARNING/SECTİON3/PREDICTION/satislar.csv")

#################### data preprocessing ####################

Months = data[["Aylar"]]
Sales = data[["Satislar"]]

#################### Train Test Split ####################

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(Months,Sales,test_size=0.33,random_state=0)

#################### Standart Scaler ####################

# from sklearn.preprocessing  import StandardScaler 

# sc =StandardScaler()

# X_train = sc.fit_transform(x_train)
# X_test = sc.fit_transform(x_test)
# Y_train = sc.fit_transform(y_train)
# Y_test = sc.fit_transform(y_test)

#################### Linear Regression ####################

#train 

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x_train,y_train)

# test

predict = lr.predict(x_test)


#################### Görselleştirme ####################

X_train = x_train.sort_index()
Y_train = y_train.sort_index()

plt.plot(X_train,Y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title("Aylara göre satışlar")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
plt.show





