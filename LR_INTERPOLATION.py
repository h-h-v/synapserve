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
#LOGISTIC REGRESSION USING INTERPOLATE
l = []
result1  = pd.DataFrame()
for i in unique:
    sf = f.loc[f['CI_Subcat']==i]
    sf_i = sf.interpolate()
    l.append(sf_i)
for i in l:
    result1 = pd.concat([result1,i])
result1.to_csv("FIXED.csv")
result1.drop_duplicates( )
result1 = result1[result1["Priority"].notna()]
print(result1.notnull().sum())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
result1["CI_Cat"]=le.fit_transform(result1["CI_Cat"])
result1["CI_Name"]=le.fit_transform(result1["CI_Name"])
result1["Category"]=le.fit_transform(result1["Category"])
x = np.array(result1[["CI_Name","CI_Cat","Category"]])
target = np.array(result1["Priority"])
y = (target<3).astype(int)
x_train,x_test,y_train,y_test = train_test_split(x,y)

from sklearn.linear_model import LogisticRegression
l = LogisticRegression()
m = l.fit(x_train,y_train)
yop = m.predict(x_test)

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

a = accuracy_score(yop,y_test)
f1=f1_score(yop,y_test)
re=recall_score(yop,y_test)
ps=precision_score(yop,y_test)
print("accuracy=",a)
print("f1_score=",f1)
print("recall_score=",re)
print("precision_score=",ps)