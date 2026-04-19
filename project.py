# Step 1: Import libraries
import pandas as pd
import numpy as np

# Step 2: Load dataset
df = pd.read_csv('weather.csv')

# Step 3: Show first 5 rows
print(df.head())

# Step 4: Basic info
print(df.info())

# Step 5: Check missing values
print(df.isnull().sum())
# Step 6: Handle missing values
df = df.dropna()

# Step 7: Confirm no missing values
print(df.isnull().sum())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])
        from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data size:", X_train.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)