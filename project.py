# Weather Dataset Preprocessing + ML Model

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('weather.csv')

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nColumn Names:")
print(df.columns)

print("\nMissing values before cleaning:")
print(df.isnull().sum())

df = df.dropna()

print("\nMissing values after cleaning:")
print(df.isnull().sum())


df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df = df.drop('Date', axis=1)


from sklearn.preprocessing import LabelEncoder

categorical_cols = ['City', 'State', 'AQI_Category']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print("\nData after encoding:")
print(df.head())


from sklearn.model_selection import train_test_split

y = df['AQI_Category']
X = df.drop('AQI_Category', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nData preprocessing completed successfully!")


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = model.score(X_test, y_test)
print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nSample Predictions:", y_pred[:5])


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import pandas as pd

lr = LogisticRegression(max_iter=200)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

print("Logistic Regression:", lr.score(X_test, y_test))
print("Decision Tree:", dt.score(X_test, y_test))
print("Random Forest:", rf.score(X_test, y_test))

lr_cv = cross_val_score(lr, X, y, cv=5)
dt_cv = cross_val_score(dt, X, y, cv=5)
rf_cv = cross_val_score(rf, X, y, cv=5)

print("LR CV:", lr_cv.mean())
print("DT CV:", dt_cv.mean())
print("RF CV:", rf_cv.mean())

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)

importance = rf.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print(feature_importance)
