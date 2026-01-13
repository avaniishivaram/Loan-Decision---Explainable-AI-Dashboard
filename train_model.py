import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# PATHS
DATA_PATH = "loan_data.csv"   # place dataset in same folder
MODEL_PATH = "model.pkl"

# LOAD DATA
df = pd.read_csv(DATA_PATH)

# DROP ID COLUMN
df.drop(columns=["Loan_ID"], inplace=True)

# HANDLE MISSING VALUES
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# TARGET
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# COLUMN TYPES
categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(exclude="object").columns

# PREPROCESSING
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# MODEL
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ]
)

# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TRAIN
model.fit(X_train, y_train)

# EVALUATE
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# SAVE MODEL
joblib.dump(model, MODEL_PATH)
print("✅ Model saved as model.pkl")
