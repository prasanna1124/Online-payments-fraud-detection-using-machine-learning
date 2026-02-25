import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# -----------------------------
# 1️⃣ LOAD DATASET
# -----------------------------

df = pd.read_csv("PS_20174392719_1491204439457_logs.csv")

# Keep required columns
df = df[['step', 'type', 'amount', 'oldbalanceOrg',
         'newbalanceOrig', 'oldbalanceDest',
         'newbalanceDest', 'isFlaggedFraud', 'isFraud']]

print("Fraud Distribution:")
print(df['isFraud'].value_counts())

# -----------------------------
# 2️⃣ FEATURES & TARGET
# -----------------------------

X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# -----------------------------
# 3️⃣ TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # VERY IMPORTANT
)

# -----------------------------
# 4️⃣ PREPROCESSING
# -----------------------------

numeric_features = [
    'step', 'amount', 'oldbalanceOrg',
    'newbalanceOrig', 'oldbalanceDest',
    'newbalanceDest', 'isFlaggedFraud'
]

categorical_features = ['type']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# -----------------------------
# 5️⃣ BALANCED RANDOM FOREST
# -----------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced',   # 🔥 VERY IMPORTANT
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# -----------------------------
# 6️⃣ TRAIN
# -----------------------------

pipeline.fit(X_train, y_train)

# -----------------------------
# 7️⃣ EVALUATE
# -----------------------------

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# -----------------------------
# 8️⃣ SAVE MODEL
# -----------------------------

joblib.dump(pipeline, "payments.pkl")

print("\n✅ Model saved as payments.pkl")