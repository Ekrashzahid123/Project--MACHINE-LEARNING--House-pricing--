import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('data.csv')

# Define features and target
x = data[['Occupation', 'Location', 'Household_Size']]
y = data['Income']

# Apply One-Hot Encoding to categorical features
encoder = OneHotEncoder(drop='first')  # Avoid dummy variable trap
encoded_features = encoder.fit_transform(x[['Occupation', 'Location']]).toarray()

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Occupation', 'Location']))

# Concatenate with numeric data
x_encoded = pd.concat([x[['Household_Size']].reset_index(drop=True), encoded_df], axis=1)

# Normalize all features (Standardization)
scaler = StandardScaler()
x_normalized = scaler.fit_transform(x_encoded)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# Visualization: Actual vs Predicted Values
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, color='blue', edgecolors='k', alpha=0.7, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Perfect Fit Line')
plt.title("Actual vs Predicted Income")
plt.xlabel("Actual Income")
plt.ylabel("Predicted Income")
plt.legend()
plt.grid(True)

# Adjust axis limits to zoom in if needed
plt.xlim(min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max()))
plt.ylim(min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max()))

plt.show()

