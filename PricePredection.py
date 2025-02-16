import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

X = np.array([
    [2, 6, 128, 2815],  
    [1, 6, 256, 3227],
    [3, 8, 128, 4000],  
    [2, 8, 256, 4500], 
    [3, 8, 128, 4300], 
    [2, 12, 256, 4500],
    [3, 8, 128, 4080], 
    [2, 8, 256, 4614],  
    [3, 8, 128, 4780],  
    [2, 12, 256, 4600],
])

Y = np.array([600, 750, 450, 600, 350, 500, 400, 550, 350, 500]).reshape(-1, 1)  # Price in USD

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)


joblib.dump(model, "price_model.pkl")
print("✅ Model saved successfully!")

new_mobile = np.array([[2, 8, 128, 4000]])  
prediction = model.predict(new_mobile)
print("New Data",prediction)