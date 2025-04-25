# Importing Libraries 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.express as px 
import plotly.graph_objects as go 
import streamlit as st 
import io 
import yfinance as yf 
import statsmodels.api as sm 
from sklearn.svm import SVR
import datetime
from datetime import date
from datetime import time,timedelta
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.layers import GRU
from keras.applications import ResNet50
from keras import layers
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title("Stock Market price prediction App Created by Maria Nadeem!!")
st.text("This App Forcast The stock Market price of selected company")
st.image("https://images.unsplash.com/photo-1563986768711-b3bde3dc821e?q=80&w=1468&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
st.sidebar.header("Select the parameters")
start_date=st.sidebar.date_input('Start Date', date(2001,1,1))
end_date=st.sidebar.date_input('End Date', date(2025,1,1))  # Updated to 2025
ticker_list=["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker=st.sidebar.selectbox("Select the Company", ticker_list)

# Fetch the data using yahoofinance library
data = yf.download(ticker, start=start_date, end=end_date)
data.reset_index(drop=True, inplace=True)
st.write('Data From', start_date, 'to', end_date)
st.write(data)

# Plot the Data 
st.header("Data Visualization plot")
st.subheader("Plot the Data")
plt.figure(figsize=(6,5))
st.write("select the **specific date** from the date range or zoom in the plot for detailed visualization and select the specific column")

# Create a dropdown for selecting which column to plot
plot_column = st.selectbox("Select column to plot", data.columns[1:])
fig=px.line(data, x="Date", y=plot_column, title=f"{plot_column} price of {ticker}", width=1000, height=600)
st.plotly_chart(fig)

# create a selection box to choose the column for forecasting
column=st.selectbox("Select the column for forecasting", data.columns[1:])
data=data[column]
st.write("Selected Data for forecasting")
st.write(data)

# Apply the ADF test to check the stationarity
st.write("#### ADF test to check the stationarity")
st.write(adfuller(data[column].dropna()[:1]<0.05)  # Added dropna() to handle missing values

# Decompose the Data and make the decomposition plot 
try:
    decomposition=seasonal_decompose(data[column].dropna(), model='additive', period=12)
    st.write("Decomposition Plot")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8))
    decomposition.observed.plot(ax=ax1)
    ax1.set_ylabel('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_ylabel('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_ylabel('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_ylabel('Residual')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Now making the decomposition plot using plotly
    st.write("Decomposition Plot using plotly")
    st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, width=1000, height=400, 
                      title="Trend", labels={"x":"Date", "y":"price"}).update_traces(line_color="green"))
    st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, width=1000, height=400, 
                      title="Seasonality", labels={"x":"Date", "y":"price"}).update_traces(line_color="blue"))
    st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, width=1000, height=400, 
                      title="Residual", labels={"x":"Date", "y":"price"}).update_traces(line_color="red"))
except ValueError as e:
    st.warning(f"Decomposition failed: {str(e)}. Not enough data points for decomposition.")

# Select the Model
models=["SARIMA","Random Forest","LSTM", "Prophet","GRU","SVM","DenseNet"]
selected_model=st.sidebar.selectbox("Select the Model for Forecasting", models)

if selected_model=="SARIMA":
    st.header("SARIMA Model")
    p=st.slider("Select the value of p", 0,5,2)
    d=st.slider("Select the value of d", 0,5,1)
    q=st.slider("Select the Value of q", 0,5,2)
    seasonal_order=st.number_input("Select the seasonal period", 0,24,12)
    
    try:
        model=sm.tsa.statespace.SARIMAX(data[column].dropna(), order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
        model=model.fit()
        
        #Summary of the model 
        st.header("Model Summary")
        st.write(model.summary())
        st.write("---")
        
        # Forecasting using SARIMA
        st.write("<p style='color:red; font-size: 50px; font-weight: bold;'>Forecasting the data with SARIMA</p>",
                 unsafe_allow_html=True)
        
        forecast_period=st.number_input("Select the Number of days to forecast", 1,365,10)
        predictions=model.get_forecast(steps=forecast_period)
        predictions_df = predictions.conf_int()
        predictions_df['Predictions'] = model.predict(start=predictions_df.index[0], end=predictions_df.index[-1])
        
        # Create future dates
        last_date = data['Date'].iloc[-1]
        prediction_dates = pd.date_range(last_date, periods=forecast_period+1)[1:]
        
        # Plotting
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data[column], name='Actual', mode='lines', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=prediction_dates, y=predictions_df['Predictions'], name='Predicted', 
                               mode='lines', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=prediction_dates, y=predictions_df.iloc[:, 0], 
                               fill=None, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=prediction_dates, y=predictions_df.iloc[:, 1], 
                               fill='tonexty', mode='lines', line=dict(width=0), showlegend=False))
        fig.update_layout(title=f'SARIMA Forecast for {ticker}', xaxis_title='Date', yaxis_title='Price',
                         width=1000, height=600)
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error in SARIMA model: {str(e)}")

elif selected_model=="Random Forest":
    st.header("Random Forest Regression")
    
    # Feature Engineering - Convert dates to numeric values
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    
    # Splitting Data into training and Testing set
    train_size=int(len(data)*0.8)
    train_data, test_data=data[:train_size], data[train_size:]
    
    # Prepare features and target
    train_X, train_y=train_data[['Days']], train_data[column]
    test_X, test_y=test_data[['Days']], test_data[column]
    
    # Train the random forest model 
    rf_model=RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(train_X, train_y)
    
    # Predictions
    train_predictions = rf_model.predict(train_X)
    test_predictions = rf_model.predict(test_X)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(train_y, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(test_y, test_predictions))
    st.write(f"Train Root Mean Squared Error: {train_rmse}")
    st.write(f"Test Root Mean Squared Error: {test_rmse}")
    
    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_data['Date'], y=test_predictions, mode='lines', name='Predicted', 
                           line=dict(color='red')))
    fig.update_layout(title=f'Random Forest Predictions for {ticker}', xaxis_title='Date', 
                     yaxis_title='Price', width=1000, height=600)
    st.plotly_chart(fig)

elif selected_model == "SVM":
    st.header("Support Vector Machine (SVM)")
    
    # Feature Engineering - Convert dates to numeric values
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[[column, 'Days']])
    
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    def create_sequences(dataset, seq_length):
        X, y = [], []
        for i in range(len(dataset) - seq_length):
            X.append(dataset[i:i + seq_length, 1])  # Using Days as feature
            y.append(dataset[i + seq_length, 0])    # Using Price as target
        return np.array(X), np.array(y)
    
    seq_length = st.slider("Select the Sequence Length", 1, 30, 10)
    train_X, train_y = create_sequences(train_data, seq_length)
    test_X, test_y = create_sequences(test_data, seq_length)
    
    # Build and train SVM model
    svm_model = SVR(kernel='rbf')
    svm_model.fit(train_X, train_y)
    
    # Predictions
    train_predictions = svm_model.predict(train_X)
    test_predictions = svm_model.predict(test_X)
    
    # Inverse scaling
    train_predictions = train_predictions.reshape(-1, 1)
    test_predictions = test_predictions.reshape(-1, 1)
    
    # Create dummy arrays for inverse transform
    train_pred_inv = np.zeros((len(train_predictions), 2))
    test_pred_inv = np.zeros((len(test_predictions), 2))
    
    train_pred_inv[:, 0] = train_predictions.flatten()
    test_pred_inv[:, 0] = test_predictions.flatten()
    
    train_predictions = scaler.inverse_transform(train_pred_inv)[:, 0]
    test_predictions = scaler.inverse_transform(test_pred_inv)[:, 0]
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(train_y, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(test_y, test_predictions))
    st.write(f"Train Root Mean Squared Error: {train_rmse}")
    st.write(f"Test Root Mean Squared Error: {test_rmse}")
    
    # Prepare dates for plotting
    train_dates = data['Date'][seq_length:train_size]
    test_dates = data['Date'][train_size+seq_length:]
    
    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_dates, y=test_predictions, mode='lines', name='Predicted', 
                           line=dict(color='red')))
    fig.update_layout(title=f'SVM Predictions for {ticker}', xaxis_title='Date', 
                     yaxis_title='Price', width=1000, height=600)
    st.plotly_chart(fig)

elif selected_model=="LSTM":
    st.header("Long Short Term Memory (LSTM)")
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[[column]].values)
    
    # Split the data
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    def create_sequences(dataset, seq_length):
        X, y = [], []
        for i in range(len(dataset) - seq_length):
            X.append(dataset[i:i + seq_length, 0])
            y.append(dataset[i + seq_length, 0])
        return np.array(X), np.array(y)
    
    seq_length = st.slider("Select the Sequence Length", 1, 30, 10)
    train_X, train_y = create_sequences(train_data, seq_length)
    test_X, test_y = create_sequences(test_data, seq_length)
    
    # Reshape for LSTM [samples, time steps, features]
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))
    
    # Build LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(train_X.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(units=1))
    
    # Compile and train
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(train_X, train_y, epochs=20, batch_size=16, verbose=1)
    
    # Predictions
    train_predictions = lstm_model.predict(train_X)
    test_predictions = lstm_model.predict(test_X)
    
    # Inverse transform
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    train_y = scaler.inverse_transform(train_y.reshape(-1, 1))
    test_y = scaler.inverse_transform(test_y.reshape(-1, 1))
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(train_y, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(test_y, test_predictions))
    st.write(f"Train Root Mean Squared Error: {train_rmse}")
    st.write(f"Test Root Mean Squared Error: {test_rmse}")
    
    # Prepare dates for plotting
    train_dates = data['Date'][seq_length:train_size]
    test_dates = data['Date'][train_size+seq_length:]
    
    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_dates, y=test_predictions.flatten(), mode='lines', 
                           name='Predicted', line=dict(color='red')))
    fig.update_layout(title=f'LSTM Predictions for {ticker}', xaxis_title='Date', 
                     yaxis_title='Price', width=1000, height=600)
    st.plotly_chart(fig)

elif selected_model=="Prophet":
    st.header("Facebook Prophet Model")
    
    # Prepare data for Prophet
    prophet_data = data[['Date', column]].copy()
    prophet_data.columns = ['ds', 'y']
    
    # Create and fit model
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(prophet_data)
    
    # Create future dataframe
    forecast_period = st.number_input("Select the Number of days to forecast", 1, 365, 30)
    future = prophet_model.make_future_dataframe(periods=forecast_period)
    
    # Forecast
    forecast = prophet_model.predict(future)
    
    # Plot forecast
    fig1 = prophet_model.plot(forecast)
    plt.title(f'Prophet Forecast for {ticker}')
    st.pyplot(fig1)
    
    # Plot components
    fig2 = prophet_model.plot_components(forecast)
    st.pyplot(fig2)
    
    # Calculate metrics on test data if available
    if len(data) > forecast_period:
        actual = data[column].values[-forecast_period:]
        predicted = forecast['yhat'].values[-forecast_period:]
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        st.write(f"RMSE on test period: {rmse}")

elif selected_model == "GRU":
    st.header("Gated Recurrent Unit (GRU)")
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[[column]].values)
    
    # Split the data
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    def create_sequences(dataset, seq_length):
        X, y = [], []
        for i in range(len(dataset) - seq_length):
            X.append(dataset[i:i + seq_length, 0])
            y.append(dataset[i + seq_length, 0])
        return np.array(X), np.array(y)
    
    seq_length = st.slider("Select the Sequence Length", 1, 30, 10)
    train_X, train_y = create_sequences(train_data, seq_length)
    test_X, test_y = create_sequences(test_data, seq_length)
    
    # Reshape for GRU [samples, time steps, features]
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))
    
    # Build GRU model
    gru_model = Sequential()
    gru_model.add(GRU(units=50, return_sequences=True, input_shape=(train_X.shape[1], 1)))
    gru_model.add(GRU(units=50))
    gru_model.add(Dense(units=1))
    
    # Compile and train
    gru_model.compile(optimizer='adam', loss='mean_squared_error')
    gru_model.fit(train_X, train_y, epochs=20, batch_size=16, verbose=1)
    
    # Predictions
    train_predictions = gru_model.predict(train_X)
    test_predictions = gru_model.predict(test_X)
    
    # Inverse transform
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    train_y = scaler.inverse_transform(train_y.reshape(-1, 1))
    test_y = scaler.inverse_transform(test_y.reshape(-1, 1))
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(train_y, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(test_y, test_predictions))
    st.write(f"Train Root Mean Squared Error: {train_rmse}")
    st.write(f"Test Root Mean Squared Error: {test_rmse}")
    
    # Prepare dates for plotting
    train_dates = data['Date'][seq_length:train_size]
    test_dates = data['Date'][train_size+seq_length:]
    
    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_dates, y=test_predictions.flatten(), mode='lines', 
                           name='Predicted', line=dict(color='red')))
    fig.update_layout(title=f'GRU Predictions for {ticker}', xaxis_title='Date', 
                     yaxis_title='Price', width=1000, height=600)
    st.plotly_chart(fig)

elif selected_model == "DenseNet":
    st.header("DenseNet Model")
    
    # Feature Engineering - Convert dates to numeric values
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Days', column]].values)
    
    # Split the data
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    def create_sequences(dataset, seq_length):
        X, y = [], []
        for i in range(len(dataset) - seq_length):
            X.append(dataset[i:i + seq_length, 0])  # Days as feature
            y.append(dataset[i + seq_length, 1])     # Price as target
        return np.array(X), np.array(y)
    
    seq_length = st.slider("Select the Sequence Length", 1, 30, 10)
    train_X, train_y = create_sequences(train_data, seq_length)
    test_X, test_y = create_sequences(test_data, seq_length)
    
    # Build DenseNet model
    model = Sequential()
    model.add(Dense(128, input_dim=seq_length, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    # Compile and train
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(train_X, train_y, epochs=50, batch_size=16, verbose=1)
    
    # Predictions
    train_predictions = model.predict(train_X)
    test_predictions = model.predict(test_X)
    
    # Inverse transform
    train_predictions = scaler.inverse_transform(np.concatenate([train_X[:, -1:], train_predictions], axis=1))[:, 1]
    test_predictions = scaler.inverse_transform(np.concatenate([test_X[:, -1:], test_predictions], axis=1))[:, 1]
    
    # Get actual values (already inverse transformed)
    train_y_actual = scaler.inverse_transform(np.concatenate([train_X[:, -1:], train_y.reshape(-1, 1)], axis=1))[:, 1]
    test_y_actual = scaler.inverse_transform(np.concatenate([test_X[:, -1:], test_y.reshape(-1, 1)], axis=1))[:, 1]
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(train_y_actual, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(test_y_actual, test_predictions))
    st.write(f"Train Root Mean Squared Error: {train_rmse}")
    st.write(f"Test Root Mean Squared Error: {test_rmse}")
    
    # Prepare dates for plotting
    train_dates = data['Date'][seq_length:train_size]
    test_dates = data['Date'][train_size+seq_length:]
    
    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_dates, y=test_predictions, mode='lines', name='Predicted', 
                           line=dict(color='red')))
    fig.update_layout(title=f'DenseNet Predictions for {ticker}', xaxis_title='Date', 
                     yaxis_title='Price', width=1000, height=600)
    st.plotly_chart(fig)

else:
    st.write("Invalid model selected.")

st.write("Model selected:", selected_model)
st.sidebar.markdown("---")

# add author name and info
st.sidebar.markdown("### Author: Maria Nadeem🎉🎊⚡")
st.sidebar.markdown("### GitHub: [GitHub](https://github.com/marianadeem755)")
st.sidebar.markdown("### Linkdin: [Linkdin Account](https://www.linkedin.com/in/maria-nadeem-4994122aa/)")
st.sidebar.markdown("### Contact: [Email](mailto:marianadeem755@gmail.com)")
st.sidebar.markdown("### Credits: [codanics](https://codanics.com/)")
st.sidebar.markdown("---")

# urls of the images
github_url = "https://img.icons8.com/fluent/48/000000/github.png"

# redirect urls
github_redirect_url = "https://github.com/marianadeem755"

# adding a footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0; 
    width: 100%;
    background-color: #f5f5f5;
    color: #000000;
    text-align: center;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="footer">Made by Maria Nadeem<a href="{github_redirect_url}"><img src="{github_url}" width="30" height="30"></a>'
             f'<a href="https://codanics.com/">Credits: https://codanics.com/</a></div>',unsafe_allow_html=True)
