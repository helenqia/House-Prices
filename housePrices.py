import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('AmesHousing.csv')
print(data.head())

# Exploratory Data Analysis
print("MISSING VALUES")
print(data.isnull().sum())  # Check missing values per column
print(data.info())  # Summary of dataset
print(data.describe())  # Statistical summary of numerical columns

'''# Plot distribution of SalePrice
plt.hist(data['SalePrice'], bins=30)
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.title('Distribution of Sale Prices')
plt.show()'''

print(data['Lot Frontage'].isnull().sum())


# Data Preprocessing
data['Lot Frontage'].fillna(data['Lot Frontage'].median(), inplace=True)
data = pd.get_dummies(data, drop_first=True)  # One-hot encode categorical variables

print(data.isnull().sum())
print("HJERERERKRNKSJNDKJNDKJSNDKJSNDKJANDKJANDKJBDNAKJBFKJABFKJ")

# Feature Selection: correlation with SalePrice
'''corr_matrix = data.corr()
corr_with_saleprice = corr_matrix['SalePrice'].sort_values(ascending=False)
print(corr_with_saleprice.head(10)) ''' # Top 10 correlated features



# Train-Test Split
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

# Calculate RMSE and R²
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print(f'Linear Regression RMSE: {rmse:.2f}')
print(f'Linear Regression R²: {r2:.2f}')







