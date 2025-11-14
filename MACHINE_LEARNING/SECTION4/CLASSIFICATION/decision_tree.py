
#################### import libraries ####################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#################### import data ####################

data = pd.read_csv("SECTION4/CLASSIFICATION/missingvalues.csv")

#################### data preprocessing ####################

data_2 = data.iloc[:,1:3]
sex = data.iloc[:,4:]

#################### missing values ####################

from sklearn.impute import SimpleImputer as Si
imputer = Si(missing_values=np.nan,strategy="mean")

age = data.iloc[:,3:4].values  #yas

imputer = imputer.fit(age)

age = imputer.transform(age)


#################### concat ####################

result_age = pd.DataFrame(data=age,index=range(22),columns=["age"])

s = pd.concat([data_2,result_age],axis=1)

#################### Train Test Split ####################

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s,sex,test_size=0.33,random_state=0)

#################### Decision Tree ####################

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion="entropy",random_state=0)

dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)


#################### Confusion Matrix ####################

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


















