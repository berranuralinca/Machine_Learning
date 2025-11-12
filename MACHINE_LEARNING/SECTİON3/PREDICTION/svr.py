
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

#################### linear regression ####################

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X,Y)

print(lr.predict([[11]]))

#################### poly regression ####################


from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=2)

x_poly = pr.fit_transform(X)

lr = LinearRegression()

lr.fit(x_poly,Y)

from sklearn.preprocessing  import StandardScaler 

sc1 =StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()

y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

from sklearn.svm import SVR

svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli)
plt.plot(x_olcekli,svr_reg.predict(x_olcekli))
plt.show()





#################### gorsellestirme ####################

# plt.scatter(X,Y)
# plt.plot(X,lr.predict(pr.fit_transform(X)))
# plt.show()


# pr = PolynomialFeatures(degree=4)

# x_poly = pr.fit_transform(X)

# lr = LinearRegression()

# lr.fit(x_poly,Y)

# plt.scatter(X,Y)
# plt.plot(X,lr.predict(pr.fit_transform(X)))

# print(lr.predict(pr.fit_transform([[11]])))


# plt.scatter(x,y,color="red")
# plt.plot(x,lr.predict(X))
# plt.show()