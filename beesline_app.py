import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from prophet import Prophet
from scipy import stats
import re
import pickle

data = pd.read_csv(r"C:\Users\USER\Downloads\cleaned_ecom_data.csv", low_memory=False)
data.set_index('Date', inplace=True)
customers = pd.read_csv(r"C:\Users\USER\Downloads\Data\Data\System 2021-2023\customers_data.csv")

# App title
st.title("Beesline ML Analytics App")
st.write("This app perform analytics using ML algos to achieve price optimization, market forecasting and customer segmentation.")
st.image("Beesline.webp", caption='Beesline', use_column_width=True)

# List of items
item_list = ['Ultrascreen Cream Invisible Sunfilter SPF 50',
             '100% Natural Lip Balm Kit',
             '3 in 1 Micellar Cleansing Water 400ml',
             'Roll On deo Frag Free',
             'Whitening Intimate Zone Routine',
             'Propolis Facial Wash',
             '4 in 1 WHITENING CLEANSER Special Offer',
             'Whitening Eye Contour Cream SPF 30',
             'Whitening Facial Cleansing Kit',
             '4 in 1 WHITENING CLEANSER']

# Dropdown menu
selected_item = st.selectbox('Select an item:', item_list)

st.write('Plot the demand function in terms of price:')
if st.button('Plot Demand'):
    value_counts = data[data['Item Name'] == selected_item]['Item Original Price'].value_counts()
    model = LinearRegression()
    model.fit(np.array(value_counts.index).reshape(-1, 1), value_counts.values)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    X = np.array(value_counts.index).reshape(-1, 1)
    plt.scatter(value_counts.index, value_counts.values, marker='o', alpha=0.7)
    plt.xlabel('Item Original Price')
    plt.yticks([])
    plt.title('Scatter Plot with Demand Function of {}'.format(selected_item))
    plt.colorbar(label='Quantity')
    best_fit_line = X * slope + intercept
    plt.plot(value_counts.index, best_fit_line, color='red', label='Best-Fitted Line')
    plt.legend()

    # Display the plot using st.pyplot()
    st.pyplot()

st.write('Predict optimal price:')    
if st.button('Optimize'):
    last_price = data[data['Item Name'] == selected_item]['Item Original Price'].iloc[-1]
    if last_price == 0.0:
        last_price = data[data['Item Name'] == selected_item]['Item Original Price'].value_counts().index[1]
    value_counts = data[data['Item Name'] == selected_item]['Item Original Price'].value_counts()
    model = LinearRegression()
    model.fit(np.array(value_counts.index).reshape(-1, 1), value_counts.values)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    if slope<=0:
        prices = np.linspace(0, 4*last_price, 1000)
        demands = model.predict(prices.reshape(-1,1))
        revenue = prices * demands
        optimal_price = prices[np.argmax(revenue)]
    else:
        optimal_price = 1.5*last_price
        
    st.write('last price: ', round(last_price, 2), '$')
    st.write('optimal price: ', round(optimal_price, 2), '$')
    st.write('% change: ', round(100*(optimal_price-last_price)/last_price, 2), '%')
    
st.write('Predict Future Product Sales: ')
if st.button('Predict'):
    product_df = data[data['Item Name'] == selected_item].copy()
    product_df['daily_sales'] = product_df['Item Original Price'] * product_df['Qty Sold']
    sales_data = product_df.groupby(['Date']).agg({'daily_sales': 'sum'}).reset_index()
    sales_data = sales_data.rename(columns={'Date': 'ds', 'daily_sales': 'y'})
    model = Prophet()
    model.fit(sales_data)
    future = model.make_future_dataframe(periods=100)
    forecast = model.predict(future)
    
    fig = model.plot(forecast)
    plt.scatter(sales_data['ds'], sales_data['y'], color='red', label='Actual')
    plt.plot(forecast['ds'], forecast['yhat'], color='blue', label='Forecast')
    plt.title('Actual vs Forecasted Daily Sales')
    plt.xlabel('Date')
    plt.ylabel('Total Sales (BRL)')
    plt.legend()
    st.pyplot()
    
    new_average = round(forecast['yhat'].tail(100).mean(),2)
    old_average = round(forecast['yhat'][:-100].mean(),2)
    if new_average >= old_average:
        st.write(f"Daily Sales for '{selected_item}' will increase from {old_average:.2f}$ ", " to ", f" {new_average:.2f}$ per day!")
    else:
        st.write(f"Daily Sales for '{selected_item}' will decrease from ${old_average:.2f}$ ", " to ", f" {new_average:.2f}$ per day!")

    
st.write('Predict Next Year Sales Overall: ')
if st.button('Predict Sales'):
    sales_data = data.groupby(['Date']).agg({'Grand Total (After Discount)': 'sum'}).reset_index()
    sales_data = sales_data.rename(columns={'Date': 'ds', 'Grand Total (After Discount)': 'y'})
    sales_data = sales_data[(np.abs(stats.zscore(sales_data['y'])) < 1.5)]
    m = Prophet()
    m.fit(sales_data)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    forecasted_prices = forecast[['ds', 'yhat']].tail(365)
    avg = forecasted_prices['yhat'].mean()
    st.write(f'On average, daily sales will attain around {round(avg,2)}$ next year!')
    
    fig = m.plot(forecast)
    plt.scatter(sales_data['ds'], sales_data['y'], color='red', label='Actual')
    plt.plot(forecast['ds'], forecast['yhat'], color='blue', label='Forecast')
    plt.title('Actual vs Forecasted Daily Sales')
    plt.xlabel('Date')
    plt.ylabel('Total Sales (BRL)')
    plt.legend()
    st.pyplot()
    

def clean_text(text):

    # Replace commas with spaces
    text = text.replace(',', ' ')

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation (excluding commas) using regex
    text =text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove extra whitespace
    text = re.sub('\s+',' ', text).strip()

    return text    

with open(r"C:\Users\USER\Downloads\tfidf_vectorizer.pkl", 'rb') as file:
    tfidf = pickle.load(file)

with open(r"C:\Users\USER\Downloads\kmeans_model.pkl", 'rb') as file:
    kmeans = pickle.load(file)
        
def transform_tfidf(text):       
    return tfidf.transform([text]).toarray().tolist()[0]
    
def predict_cluster(X, Y):
    return kmeans.predict(np.array(X+Y).reshape(1, -1))

orders = st.number_input("Enter number of orders", min_value=1, max_value=10, value=1, step=1, format="%d")
qtt = st.number_input("Enter quantity purchased", min_value=1, max_value=25, value=1, step=1, format="%d")
spend = st.number_input("Enter total spending", min_value=0.0, step=0.01, format="%f")
checked_1 = st.checkbox("Choose Bundle")
checked_2 = st.checkbox("Choose Discount")    
cat = st.text_input("Enter category description: ")


if checked_1:
    bundle = 1
bundle = 0

if checked_2:
    discount = 1
discount = 0

X = [orders, qtt, spend, bundle, discount]

cat = clean_text(cat)
Y = transform_tfidf(cat)

st.write('Predict Customer Segment: ')
if st.button('Create Persona'):
    label = predict_cluster(X, Y)
    if label == 0:
        st.image("Screenshot 2023-08-22 180716.png", caption='Cluster 0', use_column_width=True)
    else:
        st.image("Screenshot 2023-08-22 183911.png", caption='Cluster 1', use_column_width=True)