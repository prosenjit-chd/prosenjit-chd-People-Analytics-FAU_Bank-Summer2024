
import pandas as pd

# Load the dataset
file_path = 'FAU_Bank_Employee_Performance.xls'
data = pd.read_excel(file_path)

# Display the first few rows of the dataset
data.head()

# Check for missing values in the dataset
missing_data = data.isnull().sum()
missing_data

# Drop unnecessary columns
data.drop(columns=['EmpNumber'], inplace=True)

import matplotlib.pyplot as plt
import seaborn as sns

# Department-wise performance rating mean
dept_perf_mean = data.groupby('EmpDepartment')['PerformanceRating'].mean().sort_values()

# Plot department-wise performance
plt.figure(figsize=(12, 6))
sns.barplot(x=dept_perf_mean.index, y=dept_perf_mean.values, palette='plasma')
plt.title('Department-wise Employee Performance Rating')
plt.xlabel('Department')
plt.ylabel('Average Performance Rating')
plt.xticks(rotation=45)
plt.show()

from sklearn.preprocessing import LabelEncoder

# List of categorical columns
categorical_features = ['Gender', 'EducationBackground', 'MaritalStatus', 'EmpDepartment', 
                        'EmpJobRole', 'BusinessTravelFrequency', 'OverTime', 'Attrition']

# Initialize LabelEncoder
encoder = LabelEncoder()

# Apply LabelEncoder to each categorical column
for feature in categorical_features:
    data[feature] = encoder.fit_transform(data[feature])

# Display the first few rows of the dataset to verify changes
data.head()

from sklearn.ensemble import RandomForestClassifier

# Define features and target variable
features = data.drop(columns=['PerformanceRating'])
target = data['PerformanceRating']

# Initialize the Random Forest model
forest_model = RandomForestClassifier(random_state=42)

# Train the model
forest_model.fit(features, target)

# Get feature importances
importance_values = pd.Series(forest_model.feature_importances_, index=features.columns).sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 6))
sns.barplot(x=importance_values.values, y=importance_values.index, palette='plasma')
plt.title('Feature Importances in Predicting Employee Performance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Display the top important features
importance_values.head(10)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
forest_model.fit(X_train, y_train)

# Make predictions
y_pred = forest_model.predict(X_test)

# Evaluate the model
model_accuracy = accuracy_score(y_test, y_pred)
model_report = classification_report(y_test, y_pred)

print(f'Accuracy: {model_accuracy:.2f}')
print('Classification Report:')
print(model_report)

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Plot the correlation matrix
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Sort the correlation values with respect to PerformanceRating
corr_with_perf = correlation_matrix['PerformanceRating'].sort_values(ascending=False)
corr_with_perf