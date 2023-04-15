from sklearn.model_selection import train_test_split
import optuna
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 生成一个长度为100的随机时间序列
np.random.seed(0)
df = pd.Series(np.random.randn(100), index=pd.date_range('20220101', periods=100, freq='D'))
train_size = int(len(df) * 0.8)
train, test = df[0:train_size], df[train_size:len(df)]

def objective(trial):
    p = trial.suggest_int('p', 0, 5)
    d = trial.suggest_int('d', 0, 2)
    q = trial.suggest_int('q', 0, 5)
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()
    preds = model_fit.forecast(steps=len(test))
    mse = mean_squared_error(test, preds)
    return mse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

best_params = study.best_params