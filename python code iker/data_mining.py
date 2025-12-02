import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


df = pd.read_csv('C:\\Users\\JaIk194\\Desktop\\road_traffic_processed.csv')
target_col = 'credit_collection'

X = df.drop(columns=[target_col, 'case_id'])
y = df[target_col].astype(bool)  # Ensure this is boolean or 0/1

# -----------------------------------------
# 3. Handle categorical variables
# -----------------------------------------
# Identify categorical and numerical columns
# In your sample: 'resource', 'dismissal', 'vehicleClass', 'article', 'notificationType', 'lastSent', 'matricola'
# Some of these might be categorical or can be treated as numeric. You'll need domain knowledge.
# For simplicity, let's assume the following are categorical:
categorical_cols = ['resource', 'dismissal', 'vehicleClass', 'article','notificationType', 'lastSent', 'matricola']
# Some are numeric: 'article', 'points', 'total_fine_amount', etc. will be numeric.
# Adjust as needed based on your data.

# Ensure the categorical columns are indeed of type 'object' or 'category'
for col in categorical_cols:
    if col in X.columns:
        X[col] = X[col].astype('object')

# We'll use one-hot encoding for categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # keep the numerical columns as they are
)

X_transformed = preprocessor.fit_transform(X)

# -----------------------------------------
# 4. Split data into training and test sets
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42
)

# -----------------------------------------
# 5. Train a Random Forest Classifier
# -----------------------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# -----------------------------------------
# 6. Extract and display feature importances
# -----------------------------------------
import numpy as np

# The preprocessor expands columns, so we need to get the feature names
# after encoding
feature_names = []
# Extract the categorical feature names from the OneHotEncoder
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
# Get the remainder (numeric) feature names, if any
# These come after the categorical encodings in the transformed array
all_original_columns = categorical_cols + [col for col in X.columns if col not in categorical_cols]
# The ColumnTransformer with remainder='passthrough' keeps the numeric columns in the order they appear
# after the categorical ones. We need to reconstruct their names for clarity.

# The numeric columns would be those not in categorical_cols
numeric_cols = [col for col in X.columns if col not in categorical_cols]

feature_names = list(cat_feature_names) + numeric_cols

importances = clf.feature_importances_

# Sort features by importance
indices = np.argsort(importances)[::-1]  # descending order

print("Feature ranking:")
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# -----------------------------------------
# 7. Evaluate the model (optional)
# -----------------------------------------
from sklearn.metrics import classification_report

y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
