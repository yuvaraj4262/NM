import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Step 1: Load the dataset
# -----------------------------
try:
    df = pd.read_csv('customer_survey_data.csv')
except FileNotFoundError:
    print("❌ Error: 'customer_survey_data.csv' not found in the current directory.")
    exit()
except pd.errors.EmptyDataError:
    print("❌ Error: 'customer_survey_data.csv' is empty.")
    exit()

# -----------------------------
# Step 2: Basic data checks
# -----------------------------
if df.empty:
    print("❌ Error: The dataset is empty. Please check the file contents.")
    exit()

print("✅ First 5 rows of data:")
print(df.head())

print("\n🔍 Missing values per column:")
print(df.isnull().sum())

# Drop rows with missing values (or use df.fillna() if preferred)
df = df.dropna()

# -----------------------------
# Step 3: Exploratory Data Analysis
# -----------------------------
print("\n📊 Correlation matrix:")
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation with Satisfaction")
plt.show()

# -----------------------------
# Step 4: Feature & Target Setup
# -----------------------------
target = 'satisfaction'
if target not in df.columns:
    print(f"❌ Error: Target column '{target}' not found in the dataset.")
    exit()

features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

print("\n🧠 Features used for training:")
print(features)

# -----------------------------
# Step 5: Standardize and Split Data
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 6: Train Model
# -----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Step 7: Predict and Evaluate
# -----------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📈 Mean Squared Error: {mse:.2f}")
print(f"📊 R-squared Score: {r2:.2f}")

# -----------------------------
# Step 8: Feature Importance
# -----------------------------
importances = model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n🏆 Top drivers of customer satisfaction:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance for Customer Satisfaction')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
