from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LinearRegression
import streamlit as st
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

df = pd.read_csv('housing.csv', names=cols, delim_whitespace=True)

df = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'RAD', 'B', 'LSTAT', 'MEDV']]

X = df.drop('MEDV', axis=1)
y = df.MEDV

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

lr_mod = LinearRegression().fit(X_train_scaled, y_train)

st.title('Boston Housing Price')

with st.sidebar:
    st.logo('housing.png')
    with st.spinner('Loading'):
        time.sleep(5)
        st.title('Side Bar')

    st.image('house.jpeg')

CRI = st.number_input(label = 'CRIM', min_value = 0.00000, max_value = 90.0, step=0.00001)
ZN = st.number_input(label = 'ZN', min_value = 0.00000, max_value = 100.0, step=0.00001)
INDUS = st.number_input(label = 'INDUS', min_value = 0.00000, max_value = 30.0)
RAD = st.number_input(label = 'RAD', min_value = 1, max_value = 24)
B = st.number_input(label = 'B', min_value = 1.00000, max_value = 400.0, step=0.00001)
LSTAT = st.number_input(label = 'LTSTAT', min_value = 1.00000, max_value = 38.0, step=0.00001)
CHAS = st.radio('CHAS', options = [0, 1])
# Scaling Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(np.array([[CRI, ZN, INDUS, CHAS, RAD, B, LSTAT]]))
# Note this is a base model, it could be improved over time
if st.button('Predict Boston'): 
    predicted_data = lr_mod.predict(scaled_data)
    st.write(f'Your Scaled Data is {round(predicted_data[0], 2)}')
