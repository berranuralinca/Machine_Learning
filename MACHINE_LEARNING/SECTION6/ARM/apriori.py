import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from apyori import apriori

data = pd.read_csv("SECTION6/ARM/sepet.csv",header=None)

t = []
for i in range (0,7501):
    t.append([str(data.values[i,j]) for j in range (0,20)])

rule = apriori(t,min_support=0.01, min_confidence=0.2, min_lift = 3, min_length=2)

print(list(rule))
