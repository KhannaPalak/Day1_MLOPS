import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

import os

print("Current working directory:", os.getcwd())
print("File exists:", os.path.exists("data/house_data.csv"))

data = pd.read_csv('data/house_data.csv')
data.columns = data.columns.str.strip()

print(data.columns)

x = data [['area']]
y = data['price']
model = LinearRegression()
model.fit(x,y)

joblib.dump(model, 'model.pkl')

print("model trained and saved as model.pkl")