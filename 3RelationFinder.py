import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- Load data ---
f = pd.read_csv("ITSM_data.csv")

# --- Fill Priority column mode values grouped by CI_Subcat ---
f = f.sort_values(["CI_Name", "CI_Cat", "CI_Subcat"])
unique = f["CI_Subcat"].unique()
result2 = pd.DataFrame()

for subcat in unique:
    sf = f[f['CI_Subcat'] == subcat].copy()
    mode = sf["Priority"].mode()
    if not mode.empty:
        sf["Priority"].fillna(value=mode[0], inplace=True)
    result2 = pd.concat([result2, sf])

# --- Encode target variable ---
le_y = LabelEncoder()
result2["No_of_Related_Changes"] = result2["No_of_Related_Changes"].astype(str)
Y = le_y.fit_transform(result2["No_of_Related_Changes"])

# --- Encode feature columns individually ---
le_kb = LabelEncoder()
le_cat = LabelEncoder()
le_wbs = LabelEncoder()

result2["KB_number_enc"] = le_kb.fit_transform(result2["KB_number"].astype(str))
result2["Category_enc"] = le_cat.fit_transform(result2["Category"].astype(str))
result2["WBS_enc"] = le_wbs.fit_transform(result2["WBS"].astype(str))

# --- Prepare features ---
X = result2[["KB_number_enc", "Category_enc", "WBS_enc"]]

# --- Train/test split ---
x_train, x_test, y_train, y_test = train_test_split(X, Y)

# --- Train model ---
model = RandomForestClassifier()
model.fit(x_train, y_train)

# --- Evaluate model ---
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# --- Predict last row ---
last_row = X.tail(1)
pred_encoded = model.predict(last_row)
pred_decoded = le_y.inverse_transform(pred_encoded)
if pred_decoded!='nan' and pred_decoded>0:
    print("Yes Risk Possible")
else:
    print("No Risk")