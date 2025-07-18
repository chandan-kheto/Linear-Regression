
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Prepare the Data
data = {
    'area': [1000, 1500, 2000, 2500, 3000],
    'bedrooms': [2, 3, 3, 4, 4],
    'age': [5, 10, 15, 20, 25],
    'price': [50, 70, 120, 150, 180]  # Price in lakh
}
df = pd.DataFrame(data)
print(df)

# Step 2: Define X (features) and y (target)
X = df[['area', 'bedrooms', 'age']] # Multiple features
y = df['price'] # Target variable

# Step 3: Create and Train the Model
model = LinearRegression()
model.fit(X, y)

# Step 4: Make Predictions
predictions = model.predict(X)
print(f"\nPredicted Prices: {predictions}")

# Step 5: Evaluate the Model
mse = mean_squared_error(y, predictions)
print(f"\nMean Squared Error: {mse:.2f}")

# Plot actual vs predicted prices
plt.figure(figsize=(8, 5))
plt.plot(y.values, label='Actual Price', marker='o')
plt.plot(predictions, label='Predicted Price', marker='x')
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Data Point Index')
plt.ylabel('Price (in lakhs)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Check Learned Coefficients
print("\nModel Coefficients (m1, m2, m3):", model.coef_)
print("Intercept (c):", model.intercept_)
