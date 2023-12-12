# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 21:38:24 2023

@author: gyhhu
"""
#%%
import pandas as pd

#%%
train_data = pd.read_csv('MAS352FinalDataMachineLearning.csv')
test_data = pd.read_csv('MAS352FinalDataTestMachineLearning.csv')

#%%
Y_train = train_data['r3Ptstat']
X_train = train_data.drop('r3Ptstat', axis=1)

X_test = test_data.drop('name', axis=1)

(X_train.columns==X_test.columns).sum() != len(X_test.columns)

#%%
X_train=X_train.values
Y_train=Y_train.values.reshape(-1)

X_test = X_test.values

#%%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#%%
models = {
    "KNeighborsRegressor": KNeighborsRegressor(),
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "RandomForestRegressor": RandomForestRegressor()
}

predictions = {}
model_performance = {}

for name, model in models.items():
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)
    predictions[name] = preds
    
#%%
for model_name, prediction in predictions.items():
    test_data[model_name] = prediction
    
model_names = ["KNeighborsRegressor", "LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor"]
    
test_data['mlModelsPrediction'] = test_data[model_names].mean(axis=1)
    
#%%
column_names = ["name", "mlModelsPrediction", "KNeighborsRegressor", "LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor"]

Predictions2023 = test_data[column_names]

#%%
Predictions2023.to_csv("MAS352MLPredictions.csv", index=False)
