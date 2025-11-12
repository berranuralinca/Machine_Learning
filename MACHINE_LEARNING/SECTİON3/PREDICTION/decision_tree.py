


#################### import libraries ####################
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


from sklearn.tree import DecisionTreeRegressor

d_reg = DecisionTreeRegressor(random_state=0)
d_reg.fit(X,Y)

#################### gorsellestirme ####################

plt.scatter(X,Y)
plt.plot(X,d_reg.predict(X))
plt.show()
