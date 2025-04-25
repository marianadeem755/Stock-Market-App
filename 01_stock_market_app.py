# Importing Libraries 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.express as px 
import plotly.graph_objects as go 
import streamlit as st 
import yfinance as yf 
import statsmodels.api as sm 
from sklearn.svm import SVR
import datetime
from datetime import date
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.layers import GRU
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Set page config
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title("Stock Market Price Prediction App")
st.text("This App Forecasts The Stock Market Price of Selected Company")
st.image("https://images.unsplash.com/photo-1563986768711-b3bde3dc821e?q=80&w=1468&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")

# Sidebar controls
st.sidebar.header("Select the parameters")
start_date = st.sidebar.date_input('Start Date', date(2001, 1, 1))
end_date = st.sidebar.date_input('End Date', date(2025, 1, 1))
ticker_list = ["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker = st.sidebar.selectbox("Select the Company", ticker_list)

# Fetch the data
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

data = load_data(ticker, start_date, end_date)

# Display the data
st.write('Data From', start_date, 'to', end_date)
st.write(data)

# Data Visualization
st.header("Data Visualization")
st.subheader("Closing Price Over Time")
st.write("Select the specific date from the date range or zoom in the plot for detailed visualization")

# Corrected plot - using 'Close' column instead of index
fig = px.line(data, x="Date", y="Close", title=f"{ticker} Closing Price", 
              width=1000, height=600, labels={"Close": "Price ($)"})
st.plotly_chart(fig)

# Column selection for forecasting
column = st.selectbox("Select the column for forecasting", data.columns[1:])
data = data[["Date", column]]
st.write("Selected Data for forecasting")
st.write(data)

# ADF Test
st.write("#### ADF test to check the stationarity")
adf_result = adfuller(data[column].dropna())
st.write(f"ADF p-value: {adf_result[1]:.4f}")
st.write(f"Is the data stationary? {adf_result[1] < 0.05}")

# Decomposition
st.write("### Decomposition Analysis")
try:
    decomposition = seasonal_decompose(data.set_index("Date")[column], model='additive', period=12)
    
    st.write("#### Trend Component")
    fig_trend = px.line(x=decomposition.trend.index, y=decomposition.trend, 
                        title="Trend Component", labels={"x": "Date", "y": "Value"})
    st.plotly_chart(fig_trend)
    
    st.write("#### Seasonal Component")
    fig_seasonal = px.line(x=decomposition.seasonal.index, y=decomposition.seasonal, 
                          title="Seasonal Component", labels={"x": "Date", "y": "Value"})
    st.plotly_chart(fig_seasonal)
    
    st.write("#### Residual Component")
    fig_resid = px.line(x=decomposition.resid.index, y=decomposition.resid, 
                        title="Residual Component", labels={"x": "Date", "y": "Value"})
    st.plotly_chart(fig_resid)
except ValueError as e:
    st.warning(f"Could not perform decomposition: {str(e)}")

# Model selection
models = ["SARIMA", "Random Forest", "LSTM", "Prophet", "GRU", "SVM", "DenseNet"]
selected_model = st.sidebar.selectbox("Select the Model for Forecasting", models)

if selected_model == "SARIMA":
    st.header("SARIMA Model")
    col1, col2, col3 = st.columns(3)
    with col1:
        p = st.slider("Select the value of p", 0, 5, 1)
    with col2:
        d = st.slider("Select the value of d", 0, 5, 1)
    with col3:
        q = st.slider("Select the value of q", 0, 5, 1)
    
    seasonal_order = st.number_input("Select the seasonal period", 1, 24, 12)
    
    try:
        model = sm.tsa.statespace.SARIMAX(data[column], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order))
        model_fit = model.fit(disp=False)
        
        st.header("Model Summary")
        st.write(model_fit.summary())
        
        forecast_period = st.number_input("Select the number of days to forecast", 1, 365, 30)
        forecast = model_fit.get_forecast(steps=forecast_period)
        forecast_df = pd.DataFrame({
            "Date": pd.date_range(start=data["Date"].iloc[-1], periods=forecast_period+1)[1:],
            "Predicted": forecast.predicted_mean
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["Date"], y=data[column], name="Actual", mode="lines"))
        fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted"], name="Forecast", mode="lines"))
        fig.update_layout(title=f"{ticker} {column} Forecast with SARIMA", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error in SARIMA model: {str(e)}")

elif selected_model == "Random Forest":
    st.header("Random Forest Regression")
    
    # Prepare data
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    X = data[['Days']].values
    y = data[column].values
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    st.write(f"RMSE: {rmse:.2f}")
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'][:train_size], y=y_train, name="Training Data"))
    fig.add_trace(go.Scatter(x=data['Date'][train_size:], y=y_test, name="Actual Test Data"))
    fig.add_trace(go.Scatter(x=data['Date'][train_size:], y=y_pred, name="Predicted"))
    fig.update_layout(title=f"{ticker} {column} - Random Forest Forecast")
    st.plotly_chart(fig)

elif selected_model == "LSTM":
    st.header("Long Short-Term Memory (LSTM)")
    
    # Prepare data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[[column]].values)
    
    # Create sequences
    seq_length = st.slider("Select sequence length", 5, 30, 10)
    
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data)-seq_length):
            X.append(data[i:(i+seq_length), 0])
            y.append(data[i+seq_length, 0])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Build model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    
    # Predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform([y_test])
    
    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
    test_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
    st.write(f"Train RMSE: {train_rmse:.2f}")
    st.write(f"Test RMSE: {test_rmse:.2f}")
    
    # Create plot data
    train_dates = data['Date'][seq_length:train_size+seq_length]
    test_dates = data['Date'][train_size+seq_length:]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[column], name="Actual Data"))
    fig.add_trace(go.Scatter(x=train_dates, y=train_predict[:,0], name="Train Predictions"))
    fig.add_trace(go.Scatter(x=test_dates, y=test_predict[:,0], name="Test Predictions"))
    fig.update_layout(title=f"{ticker} {column} - LSTM Forecast")
    st.plotly_chart(fig)

elif selected_model == "Prophet":
    st.header("Facebook Prophet Model")
    
    # Prepare data
    prophet_data = data.rename(columns={'Date': 'ds', column: 'y'})
    
    # Create and fit model
    model = Prophet()
    model.fit(prophet_data)
    
    # Make future dataframe
    future_periods = st.number_input("Days to forecast", 30, 365, 90)
    future = model.make_future_dataframe(periods=future_periods)
    
    # Forecast
    forecast = model.predict(future)
    
    # Plot forecast
    fig1 = model.plot(forecast)
    plt.title(f"{ticker} {column} - Prophet Forecast")
    st.pyplot(fig1)
    
    # Plot components
    st.write("### Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

elif selected_model == "GRU":
    st.header("Gated Recurrent Unit (GRU)")
    
    # Prepare data (similar to LSTM)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[[column]].values)
    
    seq_length = st.slider("Select sequence length", 5, 30, 10)
    
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data)-seq_length):
            X.append(data[i:(i+seq_length), 0])
            y.append(data[i+seq_length, 0])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(scaled_data, seq_length)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Build GRU model
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(GRU(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform([y_test])
    
    train_rmse = np.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
    test_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
    st.write(f"Train RMSE: {train_rmse:.2f}")
    st.write(f"Test RMSE: {test_rmse:.2f}")
    
    train_dates = data['Date'][seq_length:train_size+seq_length]
    test_dates = data['Date'][train_size+seq_length:]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[column], name="Actual Data"))
    fig.add_trace(go.Scatter(x=train_dates, y=train_predict[:,0], name="Train Predictions"))
    fig.add_trace(go.Scatter(x=test_dates, y=test_predict[:,0], name="Test Predictions"))
    fig.update_layout(title=f"{ticker} {column} - GRU Forecast")
    st.plotly_chart(fig)

elif selected_model == "SVM":
    st.header("Support Vector Machine (SVM)")
    
    # Prepare data
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    X = data[['Days']].values
    y = data[column].values
    
    # Scale data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
    
    # Train model
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train.ravel())
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    st.write(f"RMSE: {rmse:.2f}")
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'][:train_size], y=data[column][:train_size], name="Training Data"))
    fig.add_trace(go.Scatter(x=data['Date'][train_size:], y=data[column][train_size:], name="Actual Test Data"))
    fig.add_trace(go.Scatter(x=data['Date'][train_size:], y=y_pred.flatten(), name="Predicted"))
    fig.update_layout(title=f"{ticker} {column} - SVM Forecast")
    st.plotly_chart(fig)

elif selected_model == "DenseNet":
    st.header("Dense Neural Network")
    
    # Prepare data
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    X = data[['Days']].values
    y = data[column].values
    
    # Scale data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
    
    # Build model
    model = Sequential()
    model.add(Dense(64, input_dim=1, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    st.write(f"RMSE: {rmse:.2f}")
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'][:train_size], y=data[column][:train_size], name="Training Data"))
    fig.add_trace(go.Scatter(x=data['Date'][train_size:], y=data[column][train_size:], name="Actual Test Data"))
    fig.add_trace(go.Scatter(x=data['Date'][train_size:], y=y_pred.flatten(), name="Predicted"))
    fig.update_layout(title=f"{ticker} {column} - DenseNet Forecast")
    st.plotly_chart(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Author: Maria Nadeem")
st.sidebar.markdown("### GitHub: [GitHub Profile](https://github.com/marianadeem755)")
st.sidebar.markdown("### LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/maria-nadeem-4994122aa/)")
st.sidebar.markdown("### Contact: [Email](mailto:marianadeem755@gmail.com)")

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

st.markdown(
    f'<div class="footer">'
    f'Made by Maria Nadeem | '
    f'<a href="https://github.com/marianadeem755" target="_blank">GitHub</a> | '
    f'<a href="https://codanics.com/" target="_blank">Credits: Codanics</a>'
    f'</div>',
    unsafe_allow_html=True
)
