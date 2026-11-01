import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib

st.set_page_config(
    page_title= 'Diamond Price Prediction App 💎',
    page_icon= '💎',
    layout= 'centered'
)

st.title('Diamond Price Prediction App 💎')
st.write('Enter the diamond specifications to get the predicted price from the trained machine learning model.')

# Load the Model
@st.cache_resource
def load_assets():
    model1 = joblib.load('best_diamond_model.pkl')
    scaler = joblib.load('arithmetic_col_means.pkl')

    model2 = XGBRegressor()
    model2.load_model('user_diamond_model.json')

    return model1, scaler, model2
model1, training_col_means, model2 = load_assets()

def SelectBox(name, data, placeholder, key_name):
    select_box = st.selectbox(
        name,
        data, 
        index=None,
        placeholder=placeholder,
        key=key_name
    )
    return select_box

def NumberInput(name, min, max, step, key_name):
    number_input = st.number_input(
        name, 
        min_value=min,  # Minimum Carat Value In Dataset
        max_value=max, # Maximum Carat Value In Dataset
        step=step,
        key=key_name
    )
    return number_input

# Scaling Techniques Function
def arithmetic_mean_transform(data, col_means):
    arr = np.asarray(data, dtype=float)
    return arr / col_means

# Encode the string inputs
cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

tab1 , tab2 = st.tabs(['Business Model', "Customer Model"])


with tab1:
    col1,col2 = st.columns(2)

    # Get User Inputs
    with col1:
        carat = NumberInput('Carat', 0.20, 5.01, 0.01, 'carat_model1')
        depth = NumberInput('Depth', 43.00, 79.00, 0.01, 'depth_model1')
        table = NumberInput('Table', 43.00, 95.00, 0.01, 'table_model1')

    with col2:
        x = NumberInput('X', 3.73, 10.74, 0.01, 'x_model1')
        y = NumberInput('Y', 3.68, 58.90, 0.01, 'y_model1')
        z = NumberInput('Z', 1.07, 31.80, 0.01, 'z_model1')
    size = x * y * z

    cut_options = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    cut = SelectBox('Cut', cut_options, 'Choose The Cut Type', 'cut_model1')

    color_options = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    color = SelectBox('Color', color_options, 'Choose The Color', 'color_model1')

    clarity_options = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    clarity = st.select_slider(
        'Clarity',
        clarity_options,
        value=None, 
        key= 'clarity_model1'
    )

    btn = st.button('Predict Price', key='btn_model1')

    if btn:
        if cut is None or color is None or clarity is None:
            st.info('Incomplete Data, Please Complete The Data')
        else:
            encoded_cut = cut_map[cut]
            encoded_color = color_map[color]
            encoded_clarity = clarity_map[clarity]

            user_input = pd.DataFrame(
                [[
                    carat, 
                    encoded_cut, 
                    encoded_color, 
                    encoded_clarity, 
                    depth, 
                    table, 
                    x, 
                    y, 
                    z, 
                    size]],
                columns= [
                    'carat', 
                    'cut', 
                    'color', 
                    'clarity', 
                    'depth', 
                    'table', 
                    'x',
                    'y',
                    'z',
                    'size'
                    ]                  
                )

            user_input_trans = arithmetic_mean_transform(user_input, training_col_means)

            pred_price = model1.predict(user_input_trans)[0]
            st.success(f'💎 The Predicted Diamond Price Is : {pred_price:.2f}$')


# ==================== TAB 2 ====================

with tab2:
    col1,col2 = st.columns(2)

    # Get User Inputs
    with col1:
        carat_model2 = NumberInput('Carat', 0.20, 5.01, 0.01, 'carat_model2')
        cut_options = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
        cut_model2 = SelectBox('Cut', cut_options, 'Choose The Cut Type', 'cut_model2')


    with col2:
        color_options = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
        color_model2 = SelectBox('Color', color_options, 'Choose The Color', 'color_model2')
        clarity_options = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
        clarity_model2 = st.select_slider(
            'Clarity',
            clarity_options,
            value=None,
            key= 'clarity_model2'
        )

    btn_2 = st.button('Predict Price', key='btn_model2')

    if btn_2:
        if cut_model2 is None or color_model2 is None or clarity_model2 is None:
            st.info('Incomplete Data, Please Complete The Data')
        else:
            encoded_cut_2 = cut_map[cut_model2]
            encoded_color_2 = color_map[color_model2]
            encoded_clarity_2 = clarity_map[clarity_model2]

            user_input = pd.DataFrame(
                [[
                    carat_model2, 
                    encoded_cut_2, 
                    encoded_color_2, 
                    encoded_clarity_2, 
                    ]],
                columns= [
                    'carat', 
                    'cut', 
                    'color', 
                    'clarity',
                    ]                  
                )

            pred_price_model2 = model2.predict(user_input)[0]
            st.success(f'💎 The Predicted Diamond Price Is : {pred_price_model2:.2f}$')


# Web App Link
# https://diamond-price-prediction-wep.streamlit.app/