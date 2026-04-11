import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/content/Ventilator model.csv'
data = pd.read_csv(file_path)

data.columns = ['Patients_in_ICU', 'Doctors_Available','Ventilators_Available']  

X = data[['Patients_in_ICU', 'Doctors_Available']]
y = data['Ventilators_Available']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Scaling the features
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)



D = int(input("Enter the number of patients in ICU: "))
if(D<100):
  print ("The number of ventilators available are", 100-D)
else:
  print("No ventilators are available")

# 1. Correlation Heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 2. Distribution of Target Variable (Ventilators Available)
plt.figure(figsize=(8, 6))
sns.histplot(data['Ventilators_Available'], kde=True, bins=20, color='skyblue')
plt.title('Distribution of Ventilators Available')
plt.xlabel('Number of Ventilators Available')
plt.ylabel('Frequency')
plt.show()

# 3. Feature Relationships with Target Variable
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Patients_in_ICU', y='Ventilators_Available', color='blue', label='Patients in ICU')
plt.title('Patients in ICU vs Ventilators Available')
plt.xlabel('Patients in ICU')
plt.ylabel('Ventilators Available')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Doctors_Available', y='Ventilators_Available', color='green', label='Doctors Available')
plt.title('Doctors Available vs Ventilators Available')
plt.xlabel('Doctors Available')
plt.ylabel('Ventilators Available')
plt.legend()
plt.show()

# 4. Residuals Analysis
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

plt.figure(figsize=(8, 6))
sns.histplot(train_residuals, color='blue', label='Train Residuals', kde=True, bins=30)
sns.histplot(test_residuals, color='green', label='Test Residuals', kde=True, bins=30)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.legend()
plt.show()

# 5. Feature Importance
model = pipeline.named_steps["model"]  # Extract the RandomForestRegressor
feature_importance = model.feature_importances_

plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importance, y=X.columns, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()
