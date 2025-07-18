
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = {
    'Experience': [1, 2, 3, 4, 5, 6],
    'Salary': [30000, 35000, 40000, 45000, 50000, 55000]
}

df = pd.DataFrame(data)

# Split X and y
X = df[['Experience']]  # Features (must be 2D)
y = df['Salary']        # Target

# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Visualize
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Predicted Line')
plt.xlabel('Experience (Years)')
plt.ylabel('Salary')
plt.title('Linear Regression - Salary Prediction')
plt.legend()
plt.grid(True)
plt.show()

# Print equation
print(f"Equation: Salary = {model.coef_[0]:.2f} * Experience + {model.intercept_:.2f}")
# print slope
print(f'Slope: {model.coef_[0]:.2f}') # useing f-string format to print
print(f'Intercept: {model.intercept_:.2f}')

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}") # useing f-string format to print
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

