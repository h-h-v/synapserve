import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
f = pd.read_csv("ITSM_data.csv")
for i in f:
    print(i,len(f[i].unique()))
f = f.sort_values("CI_Name")
f = f.sort_values("CI_Cat")
f = f.sort_values("CI_Subcat")
unique = f["CI_Subcat"].unique()

l = []
result2  = pd.DataFrame()
for i in unique:
    sf = f.loc[f['CI_Subcat']==i]
    mode = sf["Priority"].mode()
    sf["Priority"].fillna(value=mode)
    l.append(sf)
for i in l:
    result2 = pd.concat([result2,i])
le = LabelEncoder()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Example for 'Urgency' column
result2["Urgency"] = result2["Urgency"].astype(str)  # convert all to string first
result2["Urgency"] = le.fit_transform(result2["Urgency"])

# Do the same for all columns that cause the error, e.g.:
result2["CI_Name"] = le.fit_transform(result2["CI_Name"].astype(str))
result2["CI_Cat"] = le.fit_transform(result2["CI_Cat"].astype(str))
result2["CI_Subcat"] = le.fit_transform(result2["CI_Subcat"].astype(str))
result2["WBS"] = le.fit_transform(result2["WBS"].astype(str))
result2["Impact"] = le.fit_transform(result2["Impact"].astype(str))
result2["Category"] = le.fit_transform(result2["Category"].astype(str))
result2["KB_number"] = le.fit_transform(result2["KB_number"].astype(str))
result2["Closure_Code"] = le.fit_transform(result2["Closure_Code"].astype(str))
result2["Incident_ID"] = le.fit_transform(result2["Incident_ID"].astype(str))
result2["Status"] = le.fit_transform(result2["Status"].astype(str))
result2["Priority"] = le.fit_transform(result2["Priority"].astype(str))
result2["No_of_Reassignments"] = le.fit_transform(result2["No_of_Reassignments"].astype(str))
result2["No_of_Related_Interactions"] = le.fit_transform(result2["No_of_Related_Interactions"].astype(str))
result2["No_of_Related_Incidents"] = le.fit_transform(result2["No_of_Related_Incidents"].astype(str))
result2["No_of_Related_Changes"] = le.fit_transform(result2["No_of_Related_Changes"].astype(str))
result2["Related_Change"] = le.fit_transform(result2["Related_Change"].astype(str))
print(result2.dtypes[result2.dtypes == 'object'])
result2["Alert_Status"] = le.fit_transform(result2["Alert_Status"].astype(str))
time_cols = ["Open_Time", "Reopen_Time", "Resolved_Time", "Close_Time"]
result2["Related_Interaction"] = pd.to_numeric(result2["Related_Interaction"], errors='coerce')
result2["Handle_Time_hrs"] = pd.to_numeric(result2["Handle_Time_hrs"], errors='coerce')

for col in time_cols:
    result2[col] = pd.to_datetime(result2[col], errors='coerce')  # convert or NaT if invalid
corr_matrix = result2.corr()
print(corr_matrix)

selected_columns = ['Category', 'CI_Name', 'CI_Cat', 'CI_Subcat']
subset_df = result2[selected_columns]
correlation_matrix = subset_df.corr().abs()

print("Correlation Matrix for Selected Features:\n", correlation_matrix)

x = result2[["CI_Name", "CI_Subcat"]]

# PCA expects numeric data, make sure these columns are numeric or encoded
# If they are categorical strings, you need to encode them first
# Example: convert categories to numeric labels
from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
le2 = LabelEncoder()

x_encoded = pd.DataFrame({
    "CI_Name": le1.fit_transform(x["CI_Name"]),
    "CI_Subcat": le2.fit_transform(x["CI_Subcat"])
})

pca = PCA(n_components=1)
X_pca = pca.fit_transform(x_encoded)
print(X_pca)

x_pca = pd.DataFrame(X_pca, columns=["PC1"])
x_pca = pd.concat([x_pca, result2[["CI_Cat", "Category"]]], axis=1)
result2["Priority"] = result2["Priority"].apply(lambda x: 0 if x < 3 else 1)

# Drop rows with NaNs from both features and target together to keep alignment
df = pd.concat([x_pca, result2["Priority"]], axis=1).dropna()

# Split features and target again
X = df.drop(columns=["Priority"])
y = df["Priority"]
print(X)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

x_train, x_test, y_train, y_test = train_test_split(X, y)

rd = RandomForestClassifier(  max_depth=10,
    max_features='sqrt',
    min_samples_leaf=2,
    min_samples_split=5,
    n_estimators=200,
    random_state=42)
m = rd.fit(x_train, y_train)
yop = m.predict(x_test)

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
m = gb.fit(x_train,y_train)
yog = m.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, yop))
print(accuracy_score(y_test, yog))

import xgboost as xgb
model = xgb.XGBClassifier(
    objective='multi:softmax',        # Multiclass classification
    num_class=2,        # Number of unique classes
    eval_metric='mlogloss',           # Multiclass log loss
    use_label_encoder=False,          # Avoid deprecation warning
    random_state=42
)

# Train
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
x_encoded = pd.DataFrame(le1.fit_transform(x["CI_Name"])+le2.fit_transform(x["CI_Subcat"]))
x_pca = pd.concat([x_encoded, result2[["CI_Cat", "Category"]]], axis=1)

# Drop rows with NaNs from both features and target together to keep alignment
df = pd.concat([x_pca, result2["Priority"]], axis=1).dropna()
print(df)
# Split features and target again
# Features and target
X = df.drop(columns=["Priority"])
y = df["Priority"]

# Fix column name types
X.columns = X.columns.astype(str)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y)

rd = RandomForestClassifier(  max_depth=10,
    max_features='sqrt',
    min_samples_leaf=2,
    min_samples_split=5,
    n_estimators=200,
    random_state=42)
m = rd.fit(x_train, y_train)
yop = m.predict(x_test)

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
m = gb.fit(x_train,y_train)
yog = m.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, yop))
print(accuracy_score(y_test, yog))

import xgboost as xgb
model = xgb.XGBClassifier(
    objective='multi:softmax',        # Multiclass classification
    num_class=2,        # Number of unique classes
    eval_metric='mlogloss',           # Multiclass log loss
    use_label_encoder=False,          # Avoid deprecation warning
    random_state=42
)

# Train
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

x_encoded = pd.DataFrame(le1.fit_transform(x["CI_Name"]))
x_pca = pd.concat([x_encoded, result2[["CI_Cat", "Category"]]], axis=1)
x_pca = pd.concat([x_pca, pd.DataFrame(le2.fit_transform(x["CI_Subcat"]), columns=["CI_Subcat_encoded"])], axis=1)

# Drop rows with NaNs from both features and target together to keep alignment
df = pd.concat([x_pca, result2["Priority"]], axis=1).dropna()
print(df)
# Split features and target again
# Features and target
X = df.drop(columns=["Priority"])
y = df["Priority"]

# Fix column name types
X.columns = X.columns.astype(str)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y)

rd = RandomForestClassifier(  max_depth=10,
    max_features='sqrt',
    min_samples_leaf=2,
    min_samples_split=5,
    n_estimators=200,
    random_state=42)
m = rd.fit(x_train, y_train)
yop = m.predict(x_test)

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
m = gb.fit(x_train,y_train)
yog = m.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, yop))
print(accuracy_score(y_test, yog))

import xgboost as xgb
model = xgb.XGBClassifier(
    objective='multi:softmax',        # Multiclass classification
    num_class=2,        # Number of unique classes
    eval_metric='mlogloss',           # Multiclass log loss
    use_label_encoder=False,          # Avoid deprecation warning
    random_state=42
)

# Train
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
import pandas as pd

# Create a DataFrame comparing true and predicted values
comparison_df = pd.DataFrame({
    'True Label': y_test,
    'Predicted Label (RF)': yop,
    'Predicted Label (GB)': yog
})

# Filter only misclassified rows
misclassified_rf = comparison_df[comparison_df['True Label'] != comparison_df['Predicted Label (RF)']]
misclassified_gb = comparison_df[comparison_df['True Label'] != comparison_df['Predicted Label (GB)']]
print("Random Forest mispredictions:")
print(misclassified_rf)

print("\nGradient Boosting mispredictions:")
print(misclassified_gb)
print(len(misclassified_gb))
print(len(misclassified_rf))