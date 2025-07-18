# 🏠 Linear Regression: Predicting House Price Based on Area

This mini project demonstrates how to use **Linear Regression** to predict the **price of a house** based on its **area** (in square feet). It’s a beginner-friendly machine learning project using Python and scikit-learn.

---

## 📌 Project Overview

- **Problem**: Predict house prices using area (single feature).
- **Type**: Regression
- **Model**: Linear Regression
- **Tool**: Scikit-learn (`sklearn`)
- **Goal**: Fit a straight line to predict price for any given area.

---

## 🧠 Dataset Used

```python
data = {
    'area': [1000, 1500, 2000, 2500, 3000],
    'price': [50, 75, 120, 150, 180]
}
Area (sq. ft)	Price (in Lakhs ₹)
1000	50
1500	75
2000	120
2500	150
3000	180

🔧 Steps Followed
✅ Step 1: Load and Prepare the Data
import pandas as pd

data = {
    'area': [1000, 1500, 2000, 2500, 3000],
    'price': [50, 75, 120, 150, 180]
}
df = pd.DataFrame(data)

✅ Step 2: Train the Linear Regression Model
from sklearn.linear_model import LinearRegression

X = df[['area']]  # Features (2D)
y = df['price']   # Target (1D)

model = LinearRegression()
model.fit(X, y)

✅ Step 3: Predict and Evaluate
predicted_price = model.predict([[2800]])
print(f"Predicted Price for 2800 sq. ft: ₹{predicted_price[0]:.2f} Lakhs")

✅ Step 4: Visualize the Result
import matplotlib.pyplot as plt

plt.scatter(df['area'], df['price'], color='blue', label='Actual Data')
plt.plot(df['area'], model.predict(X), color='red', label='Regression Line')
plt.xlabel("Area (sq. ft)")
plt.ylabel("Price (Lakhs ₹)")
plt.title("Linear Regression - Area vs Price")
plt.legend()
plt.grid(True)
plt.show()

📈 Output Example
Predicted Price for 2800 sq. ft: ₹166.80 Lakhs
A straight red line is fit through the data points, showing the trend of increasing house price with increasing area.

✅ What You Learn from This Project
How Linear Regression works on real-world data

How to fit a regression model using sklearn

How to make predictions for unseen values

How to visualize data and regression line

📚 Tech Stack
Pytho, pandas, matplotlib, scikit-learn
