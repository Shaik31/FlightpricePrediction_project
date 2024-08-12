
import pandas as pd
import pickle
import streamlit as st
import numpy as np

import warnings
warnings.filterwarnings('ignore')

model = pickle.load(open('xgbmodel.pkl','rb'))


st.title('Flight Price Prediction using Machine Learning')
st.subheader('User Input Parameters')

def user_input_paramters():
    airline = st.selectbox('Airline? \nselect one from the below list\n',['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo','Air_India'])
    source_city = st.selectbox('select the source city ',['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
    departure_time = st.selectbox('select the departure time from below list',['Evening', 'Early_Morning', 'Morning', 'Afternoon', 'Night',
       'Late_Night'])
    stops = st.selectbox('select the stops you would like to enjoy',['zero', 'one', 'two_or_more'])
    arrival_time = st.selectbox('what is the arrival time of your flight',['Night', 'Morning', 'Early_Morning', 'Afternoon', 'Evening',
       'Late_Night'])
    destination_city = st.selectbox('select the destination city ',['Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Delhi'])
    classtype = st.selectbox('select the class type from below',['Economy', 'Business'])
    duration = st.number_input('enter the duration of your journey here, Eg: "if its 2 hrs 30 mins" enter "2.30"')
    days_left = st.number_input('enter the number of days left for your journey ')
    data = {'airline': airline,
            'source_city':source_city,
            'departure_time': departure_time,
            'stops':stops,
            'arrival_time':arrival_time,
            'destination_city':destination_city,
            'class':classtype,
            'duration':duration,
            'days_left':days_left }
    features = pd.DataFrame(data,index=[0])
    return features

df = user_input_paramters()
st.subheader('User Input Parameters as a DataFrame')
st.write(df)
x = df

bt = st.button('Click here to Predict the price')
if bt:
    pred = model.predict(x).round(2)
    formatted_amounts = [f"₹{x:,.2f}" for x in pred]
    for amount in formatted_amounts:
        st.write("The Predicted Price of the flight is:",amount)