import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv("Walmart.csv")

# ==============================
# 2. DATE FIX
# ==============================
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# ==============================
# 3. SORT DATA
# ==============================
df = df.sort_values('Date')

# ==============================
# 4. FEATURE ENGINEERING
# ==============================
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek

# IMPORTANT: Use correct column name
df['Lag_1'] = df['Weekly_Sales'].shift(1)

# Remove NaN values
df = df.dropna()

# ==============================
# 5. FEATURES & TARGET
# ==============================
features = ['Year', 'Month', 'Day', 'DayOfWeek', 'Lag_1']
X = df[features]
y = df['Weekly_Sales']

# ==============================
# 6. TRAIN-TEST SPLIT
# ==============================
train_size = int(len(df) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

# ==============================
# 7. MODEL
# ==============================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==============================
# 8. PREDICTION
# ==============================
y_pred = model.predict(X_test)

# ==============================
# 9. EVALUATION
# ==============================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)

# ==============================
# 10. VISUALIZATION
# ==============================
plt.figure()

plt.plot(df['Date'][train_size:], y_test.values, label='Actual')
plt.plot(df['Date'][train_size:], y_pred, label='Predicted')

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Forecast")
plt.legend()

plt.show()

# ==============================
# 11. FUTURE PREDICTION
# ==============================
last = df.iloc[-1]

future = pd.DataFrame({
    'Year': [last['Year']],
    'Month': [last['Month']],
    'Day': [last['Day'] + 1],
    'DayOfWeek': [(last['DayOfWeek'] + 1) % 7],
    'Lag_1': [last['Weekly_Sales']]
})

prediction = model.predict(future)

print("Next Day Sales:", prediction[0])