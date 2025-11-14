import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('SECTION5/CLUSTERING/musteriler.csv')

data = veriler.iloc[:,2:].values


from sklearn.cluster import KMeans

km = KMeans(n_clusters=3,init="k-means++")
km.fit(data)
print(km.cluster_centers_)

results =[]

for i in range(1,11):
    km = KMeans(n_clusters=i,init="k-means++",random_state=123)
    km.fit(data)
    results.append(km.inertia_)
    
plt.plot(range(1,11),results)



km = KMeans(n_clusters=3,init="k-means++")
results = km.fit_predict(data)
print(km.cluster_centers_)

plt.scatter(data[results==0,0],data[results==0,1],color="r") # 0=x ekseni,1=y ekseni
plt.scatter(data[results==1,0],data[results==1,1],color="g")
plt.scatter(data[results==2,0],data[results==2,1],color="b")
plt.show()