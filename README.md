# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL FOR EV DATA
### Date: 
### Developed by: THANJIYAPPAN K
### Register Number: 212222240108

### AIM:
To implement the SARIMA model using Python for time series analysis on EV data.

### ALGORITHM:
1. Explore the Dataset
   - Load the EV dataset and perform initial exploration, focusing on the `year` and `value` columns. Plot the time series to visualize trends.

2. Check for Stationarity of Time Series  
   - Plot the time series data and apply the Augmented Dickey-Fuller (ADF) test to check for stationarity.

3.Determine SARIMA Model Parameters (p, d, q, P, D, Q, m)
  
4. Fit the SARIMA Model
  
5. Make Time Series Predictions and Auto-fit the SARIMA Model 
   
6. Evaluate Model Predictions
   

### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('ev.csv')

# Convert 'year' column to datetime and set it as the index
data['year'] = pd.to_datetime(data['year'], format='%Y')
data.set_index('year', inplace=True)

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['value'], label='EV Data')
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('EV Data Time Series')
plt.legend()
plt.show()

# Function to perform ADF test
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'    {key}: {value}')

# Check stationarity
check_stationarity(data['value'])

# Plot ACF and PACF to determine parameters
plot_acf(data['value'])
plt.title('ACF Plot')
plt.show()

plot_pacf(data['value'])
plt.title('PACF Plot')
plt.show()

# SARIMA model parameters (example values; adjust based on ACF/PACF insights)
p, d, q = 1, 1, 1   # Non-seasonal parameters
P, D, Q, m = 1, 1, 1, 12  # Seasonal parameters (assuming annual seasonality with monthly data)

# Split the data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data['value'][:train_size], data['value'][train_size:]

# Fit SARIMA model
sarima_model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m))
sarima_result = sarima_model.fit()

# Make predictions
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE for evaluation
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()
```

### OUTPUT:

Consumption time series:

![image](https://github.com/user-attachments/assets/82eaf511-be39-4fba-bbc6-f21f6b543d97)
![image](https://github.com/user-attachments/assets/0187fa85-5ead-4283-aabf-69525da1023e)


Autocorrelation:

![image](https://github.com/user-attachments/assets/8cc24483-ebb9-4330-9825-1a2cb8fffe56)



Partial Autocorrelation:

![image](https://github.com/user-attachments/assets/c233476a-bbec-4be0-8f01-dc3f03a2c1af)

![image](https://github.com/user-attachments/assets/7ca3b957-c4f4-408d-8f02-9466fa7f9529)


SARIMA Model Prediction:

![image](https://github.com/user-attachments/assets/b091065e-692f-4d8a-862d-afc99c9552eb)



### RESULT:
Thus the program using SARIMA model is executed successfully.
