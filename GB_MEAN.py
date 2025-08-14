import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
f = pd.read_csv("ITSM_data.csv")
for i in f:
    print(i,len(f[i].unique()))
f = f.sort_values("CI_Name")
f = f.sort_values("CI_Cat")
f = f.sort_values("CI_Subcat")
unique = f["CI_Subcat"].unique()
#LOGISTIC REGRESSION USING MEAN
l = []
result2  = pd.DataFrame()
for i in unique:
    sf = f.loc[f['CI_Subcat']==i]
    mean = sf["Priority"].mean()
    sf["Priority"].fillna(value=mean)
    l.append(sf)
for i in l:
    result2 = pd.concat([result2,i])
print(result2)
result2.to_csv("FIXED.csv")
result2.drop_duplicates( )
result2 = result2[result2["Priority"].notna()]
print(result2.notnull().sum())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
result2["CI_Subcat"]=le.fit_transform(result2["CI_Subcat"])
result2["CI_Cat"]=le.fit_transform(result2["CI_Cat"])
result2["CI_Name"]=le.fit_transform(result2["CI_Name"])
result2["Category"]=le.fit_transform(result2["Category"])
x = np.array(result2[["CI_Name","CI_Cat","CI_Subcat","Category"]])
target = np.array(result2["Priority"])
y = (target>4).astype(int)
x_train,x_test,y_train,y_test = train_test_split(x,y)

from sklearn.ensemble import GradientBoostingClassifier
l = GradientBoostingClassifier()
m = l.fit(x_train,y_train)
yop = m.predict(x_test)

from sklearn.metrics import accuracy_score

a = accuracy_score(yop,y_test)
print(a)