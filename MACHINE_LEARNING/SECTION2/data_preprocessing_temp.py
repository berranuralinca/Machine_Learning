
#################### import libraries ####################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#################### import data ####################

data = pd.read_csv("SECTION2/missingvalues.csv")

#################### data preprocessing ####################

height_weight = data[['boy','kilo']]
print(height_weight)

#################### missing values ####################

from sklearn.impute import SimpleImputer as Si
imputer = Si(missing_values=np.nan,strategy="mean")

age = data.iloc[:,3:4].values  #yas

imputer = imputer.fit(age)

age = imputer.transform(age)

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

result_age = pd.DataFrame(data=age,index=range(22),columns=["age"])

result_sex = pd.DataFrame(data=sex_le,index=range(22),columns=["sex"])

s = pd.concat([result_country,result_height_weight],axis=1)

s2 = pd.concat([s,result_age],axis=1)

s3 = pd.concat([s2,result_sex],axis=1)

#################### Train Test Split ####################

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s2,result_sex,test_size=0.33,random_state=0)

#################### Standart Scaler ####################

from sklearn.preprocessing  import StandardScaler 

sc =StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
















