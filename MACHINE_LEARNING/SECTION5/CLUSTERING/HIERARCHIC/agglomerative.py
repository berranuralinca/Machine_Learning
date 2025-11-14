import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('SECTION5/CLUSTERING/musteriler.csv')

data = veriler.iloc[:,2:].values


from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3,metric="euclidean",linkage="ward")
cluster = ac.fit_predict(data)
print(cluster)


plt.scatter(data[cluster==0,0],data[cluster==0,1],color="r") # 0=x ekseni,1=y ekseni
plt.scatter(data[cluster==1,0],data[cluster==1,1],color="g")
plt.scatter(data[cluster==2,0],data[cluster==2,1],color="b")
plt.show()

results =[]

# dendogram

from scipy.cluster import hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(data,method="ward"))
plt.show()