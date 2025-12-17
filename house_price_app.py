import streamlit as st

import numpy as np
import pandas as pd
import plotly.express as px

def generate_house_data(n_samples=100):
    np.random.seed(50)
    size = np.random.normal(1500, 50, n_samples)
    price = size * 50 + np.random.normal(0, 50 , n_samples)
    return pd.DataFrame({'Size': size, 'Price': price})

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def train_model():
    df = generate_house_data(n_samples=100)
    X = df[['Size']]
    Y = df['Price']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    model=LinearRegression()
    model.fit(X_train, Y_train)
    
    return model

def main():
    st.title("Simple Linear Regression House Prediction App")
    
    st.write("Put in your house size to predict the price.")
    
    model = train_model()
    
    size = st.number_input("Enter house size (in square feet):", min_value=50, max_value=10000, value=1500)
    
    if st.button("Predict Price"):
        predicted_price = model.predict([[size]])
        st.success(f"Estimated price : ${predicted_price[0]:,.2f}")
        
        df = generate_house_data()
        
        fig = px.scatter(df, x='Size', y='Price', title='House Size vs Price')
        fig.add_scatter(x=[size], y=predicted_price[[0]],
                mode='markers', 
                marker=dict(color='red', size=15), 
                name='Prediction')
        
        st.plotly_chart(fig)
        
if __name__ == "__main__":
    main()
            
        
    