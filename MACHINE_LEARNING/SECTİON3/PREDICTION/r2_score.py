
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#################### import data ####################

data = pd.read_csv("SECTİON3/PREDICTION/maaslar.csv")

#################### slice ####################

x = data.iloc[:,1:2]

y = data.iloc[:,2:]

#################### array dönüşümü ####################
X = x.values
Y = y.values

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(x,y,color="red")
y_pred = rf_reg.predict(X)
plt.show()

from sklearn.metrics import r2_score

print(r2_score(Y,y_pred))
