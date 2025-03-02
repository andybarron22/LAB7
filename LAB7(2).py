import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import openpyxl


file_path = "AmesHousing.xlsx"
df = pd.read_excel(file_path)


selected_features = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF', 'Full Bath', 'Year Built']
target = 'SalePrice'

df = df[selected_features + [target]].dropna()


X = df[selected_features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)


st.title('Ames Housing Price Prediction')


st.sidebar.header('Input House Features')

def user_input_features():
    Overall_Qual = st.sidebar.slider('Overall Quality', 1, 10, 5)
    Gr_Liv_Area = st.sidebar.slider('Above Ground Living Area (sq ft)', 500, 5000, 1500)
    Garage_Cars = st.sidebar.slider('Garage Cars', 0, 4, 2)
    Total_Bsmt_SF = st.sidebar.slider('Total Basement Size (sq ft)', 0, 3000, 1000)
    Full_Bath = st.sidebar.slider('Full Bathrooms', 0, 4, 2)
    Year_Built = st.sidebar.slider('Year Built', 1800, 2023, 2000)
   
    data = {
        'Overall Qual': Overall_Qual,
        'Gr Liv Area': Gr_Liv_Area,
        'Garage Cars': Garage_Cars,
        'Total Bsmt SF': Total_Bsmt_SF,
        'Full Bath': Full_Bath,
        'Year Built': Year_Built
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()


st.subheader('User Input Parameters')
st.write(input_df)


prediction = model.predict(input_df)


st.subheader('Predicted House Price ($)')
st.write(f"${prediction[0]:,.2f}")
