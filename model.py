import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# prep data 
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
y = df['value']
train_size = int(len(y) * 0.8)
y_train, y_test = y[:train_size], y[train_size:]

arima_model = ARIMA(y_train, order=(5,1,0))  # will prob need to change (p,d,q) later
res = arima_model.fit()

forecast = res.forecast(steps=len(y_test))
arima_full_pred = res.predict(start=1, end=len(y_train)-1, typ='levels')
arima_resid = y_train[1:] - arima_full_pred

def create_lag_features(series, lags=[1,2,3]):
    df_feat = pd.DataFrame()
    for lag in lags:
        df_feat[f'lag_{lag}'] = series.shift(lag)
    return df_feat

X_rf = create_lag_features(y_train).dropna()
y_rf = arima_resid.loc[X_rf.index]  # aligns residuals with features

rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_rf, y_rf)

X_test_rf = create_lag_features(y).iloc[train_size:]
X_test_rf = X_test_rf[:len(y_test)]
rf_resid_pred = rf.predict(X_test_rf)

final_pred = forecast.values + rf_resid_pred

mse = mean_squared_error(y_test[:len(final_pred)], final_pred)
print(f"esnemble: {mse:.4f}")