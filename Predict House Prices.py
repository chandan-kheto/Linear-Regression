
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Load the data
data = {
    'area': [1000, 1500, 2000, 2500, 3000],
    'price': [50, 75, 120, 150, 180]
}
df = pd.DataFrame(data)
print(df)

# Step 2: Split features (X) and target (y)
X = df[['area']]   # Features (area)
y = df['price']    # Target (price)

# Step 3: Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)
print(f"Predicted Prices: {y_pred}")
print(f"Actual Prices: {y_test.tolist()}")

# Step 6: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Optional: Visualize the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Prediction Line')
plt.xlabel('Area (sqft)')
plt.ylabel('Price (₹ lakh)')
plt.title('House Price Prediction')
plt.legend()
plt.show()
