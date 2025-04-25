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
end_date=st.sidebar.date_input('End Date', date(2025,1,1))
ticker_list=["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker=st.sidebar.selectbox("Select the Company", ticker_list)
# Fetch the data using yahoofinance library
data = yf.download(ticker, start=start_date, end=end_date)
data.insert(0,"Date", data.index, True)
data.reset_index(drop=True, inplace=True)
st.write('Data From', start_date, 'to', end_date)
st.write(data)
# Plot the Data 
st.header("Data Visualization plot")
st.subheader("Plot the Data")
plt.figure(figsize=(6,5))
st.write("select the **specific date** from the date range or zoom in the plot for detailed visualization and select the specific column")

# Fix: Use 'Close' column instead of data.index for y-axis
fig=px.line(data, x="Date", y='Close', title="Closing price of the stock", width=1000, height=600)
st.plotly_chart(fig)

# create a selection box to choose the column for forecasting
column=st.selectbox("Select the column for forecasting", data.columns[1:])
data=data[["Date", column]]
st.write("Selected Data for forecasting")
st.write(data)

# Apply the ADF test to check the stationarity
st.write("#### ADF test to check the stationarity")
result = adfuller(data[column])
st.write(f"ADF Statistic: {result[0]}")
st.write(f"p-value: {result[1]}")
st.write(f"Is data stationary (p-value < 0.05): {result[1] < 0.05}")

# Decompose the Data and make the decomposition plot 
try:
    decomposition=seasonal_decompose(data[column], model='additive', period=12)
    
    # Create Matplotlib plot
    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(decomposition.observed)
    plt.title('Observed')
    plt.subplot(412)
    plt.plot(decomposition.trend)
    plt.title('Trend')
    plt.subplot(413)
    plt.plot(decomposition.seasonal)
    plt.title('Seasonality')
    plt.subplot(414)
    plt.plot(decomposition.resid)
    plt.title('Residuals')
    plt.tight_layout()
    st.pyplot(plt)
    
    # Now making the decomposition plot using plotly
    st.write("Decomposition Plot using plotly")
    st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, width=1000, height=400, title="Trend", labels={"x":"Date", "y":"price"}).update_traces(line_color="green"))
    st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, width=1000, height=400, title="Seasonality", labels={"x":"Date", "y":"price"}).update_traces(line_color="blue"))
    st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, width=1000, height=400, title="Residual", labels={"x":"Date", "y":"price"}).update_traces(line_color="red"))
except Exception as e:
    st.error(f"Error in decomposition: {e}")
    st.info("Try selecting a different column or date range with more data points.")

# Select the Model
models=["SARIMA","Random Forest","LSTM", "Prophet","GRU","SVM","DenseNet"]
selected_model=st.sidebar.selectbox("Select the Model for Forecasting", models)

# Define forecast_period outside the models 
forecast_period = st.sidebar.number_input("Select the Number of days to forecast", 1, 365, 10, key="forecast_period")

if selected_model=="SARIMA":
    st.header("SARIMA Model")
    p=st.slider("Select the value of p", 0, 5, 2)
    d=st.slider("Select the value of d", 0, 5, 1)
    q=st.slider("Select the value of q", 0, 5, 2)
    seasonal_order=st.number_input("Select the seasonal period", 0, 24, 12)
    
    # Wrap SARIMA in try-except to handle potential errors
    try:
        model=sm.tsa.statespace.SARIMAX(data[column], order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
        # Train the model 
        model=model.fit()
        # Summary of the model 
        st.header("Model Summary")
        st.text(str(model.summary()))
        st.write("---")
        
        # Forecasting using SARIMA
        st.markdown("<p style='color:red; font-size: 30px; font-weight: bold;'>Forecasting with SARIMA</p>", unsafe_allow_html=True)
        
        predictions=model.get_prediction(start=len(data), end=len(data)+forecast_period-1)
        predictions=predictions.predicted_mean
        last_date = pd.to_datetime(data["Date"].iloc[-1])
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(predictions), freq="D")
        
        predictions_df = pd.DataFrame({'Date': future_dates, 'predicted_mean': predictions})
        st.write('Predictions', predictions_df)
        st.write("Actual Data", data)
        st.write("---")
        
        # make the plotly plot
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=data["Date"], y=data[column], name="Actual", mode="lines", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=predictions_df["Date"], y=predictions_df["predicted_mean"], name="Predicted", mode="lines", line=dict(color='red')))
        fig.update_layout(title="Actual vs Predicted", xaxis_title="Date", yaxis_title="Value", width=1000, height=400)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error in SARIMA model: {e}")
        st.info("Try adjusting the parameters or selecting a different column.")
        
elif selected_model=="Random Forest":
    st.header("Random Forest Regression")
    
    try:
        # Convert dates to numerical features for Random Forest
        data['Date_ordinal'] = pd.to_datetime(data['Date']).map(datetime.datetime.toordinal)
        
        # Splitting Data into training and Testing set
        train_size=int(len(data)*0.8)
        train_data, test_data=data[:train_size], data[train_size:]
        
        # Feature Engineering 
        train_X, train_y=train_data[['Date_ordinal']], train_data[column]
        test_X, test_y=test_data[['Date_ordinal']], test_data[column]
        
        # Train the random forest model 
        n_estimators = st.slider("Number of estimators", 10, 500, 100)
        max_depth = st.slider("Maximum depth", 1, 50, 10)
        
        rf_model=RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf_model.fit(train_X, train_y)
        
        # Predictions
        predictions=rf_model.predict(test_X)
        
        # Calculate the mean_squared_error
        mse=mean_squared_error(test_y, predictions)
        rmse=np.sqrt(mse)
        st.write(f"Root Mean Squared Error: {rmse}")
        
        # Future predictions
        last_date_ordinal = data['Date_ordinal'].iloc[-1]
        future_dates_ordinal = np.array([last_date_ordinal + i for i in range(1, forecast_period + 1)]).reshape(-1, 1)
        future_predictions = rf_model.predict(future_dates_ordinal)
        
        # Convert ordinal dates back to datetime
        future_dates = [datetime.datetime.fromordinal(int(x)) for x in future_dates_ordinal.flatten()]
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted': future_predictions})
        st.write("Future Predictions:", future_df)
        
        # Plot the data
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test_data["Date"], y=predictions, mode='lines', name='Test Predictions', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Predicted"], mode='lines', name='Future Predictions', line=dict(color='red')))
        fig.update_layout(title="Actual vs Predicted (Random Forest)", xaxis_title="Date", yaxis_title="Price", width=1000, height=400)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error in Random Forest model: {e}")
        st.info("Try selecting a different column or date range with more data points.")

elif selected_model == "SVM":
    st.header("Support Vector Machine (SVM)")
    
    try:
        # Scale the Data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[column].values.reshape(-1, 1))
        train_size = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
        
        def create_sequences(dataset, seq_length):
            X, y = [], []
            for i in range(len(dataset) - seq_length):
                X.append(dataset[i:i + seq_length, 0])
                y.append(dataset[i + seq_length, 0])
            return np.array(X), np.array(y)
        
        seq_length = st.slider("Select the Sequence Length", 1, 30, 10, key="svm_seq")
        
        # Check if we have enough data
        if len(train_data) <= seq_length or len(test_data) <= seq_length:
            st.error("Not enough data for the selected sequence length. Try reducing the sequence length.")
        else:
            train_X, train_y = create_sequences(train_data, seq_length)
            test_X, test_y = create_sequences(test_data, seq_length)
            
            # Reshape train_X and test_X for SVM
            train_X = train_X.reshape(-1, seq_length)
            test_X = test_X.reshape(-1, seq_length)
            
            # Build and train SVM model
            kernel = st.selectbox("Select kernel", ["linear", "poly", "rbf", "sigmoid"], index=2)
            C = st.slider("Regularization parameter (C)", 0.01, 10.0, 1.0)
            
            svm_model = SVR(kernel=kernel, C=C)
            svm_model.fit(train_X, train_y)
            
            # Predict the values
            train_predictions = svm_model.predict(train_X)
            test_predictions = svm_model.predict(test_X)
            
            # Reshape for inverse transformation
            train_predictions = train_predictions.reshape(-1, 1)
            test_predictions = test_predictions.reshape(-1, 1)
            
            # Inverse transform to get original scale
            train_predictions = scaler.inverse_transform(train_predictions)
            test_predictions = scaler.inverse_transform(test_predictions)
            
            # Actual values for comparison (accounting for sequence length)
            actual_train = scaler.inverse_transform(train_data[seq_length:])
            actual_test = scaler.inverse_transform(test_data[seq_length:])
            
            # Calculate the metrics
            train_mse = mean_squared_error(actual_train, train_predictions)
            train_rmse = np.sqrt(train_mse)
            test_mse = mean_squared_error(actual_test, test_predictions)
            test_rmse = np.sqrt(test_mse)
            
            st.write(f"Train Root Mean Squared Error: {train_rmse}")
            st.write(f"Test Root Mean Squared Error: {test_rmse}")
            
            # Prepare for plotting
            train_dates = data["Date"][seq_length:train_size]
            test_dates = data["Date"][train_size+seq_length:len(data)]
            
            # Future predictions
            if len(test_data) >= seq_length:
                # Use the last sequence from the data
                last_sequence = scaled_data[-seq_length:].reshape(1, -1)
                
                # Make future predictions one step at a time
                future_preds = []
                current_seq = last_sequence.copy()
                
                for _ in range(forecast_period):
                    # Predict next value
                    next_pred = svm_model.predict(current_seq)
                    future_preds.append(next_pred[0])
                    
                    # Update sequence for next prediction
                    current_seq = np.append(current_seq[:, 1:], [[next_pred[0]]], axis=1)
                
                # Convert predictions back to original scale
                future_preds = np.array(future_preds).reshape(-1, 1)
                future_preds = scaler.inverse_transform(future_preds)
                
                # Create future dates
                last_date = pd.to_datetime(data["Date"].iloc[-1])
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_period)
                
                future_df = pd.DataFrame({'Date': future_dates, 'Predicted': future_preds.flatten()})
                st.write("Future Predictions:", future_df)
                
                # Plot the data
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=test_dates, y=test_predictions.flatten(), mode='lines', name='Test Predictions', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Predicted"], mode='lines', name='Future Predictions', line=dict(color='red')))
                fig.update_layout(title='Actual vs Predicted (SVM)', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
                st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error in SVM model: {e}")
        st.info("Try adjusting the sequence length or selecting a different column.")
        
elif selected_model=="LSTM":
    st.header("Long Short Term Memory (LSTM)")
    
    try:
        # Scale the Data
        scaler=MinMaxScaler(feature_range=(0,1))
        scaled_data=scaler.fit_transform(data[column].values.reshape(-1,1))
        
        # Split the Data into Training and Testing Data 
        train_size=int(len(scaled_data)*0.8)
        train_data, test_data=scaled_data[:train_size], scaled_data[train_size:]
        
        def create_sequences(dataset, seq_length):
            X, y=[],[]
            for i in range(len(dataset)-seq_length):
                X.append(dataset[i:i+seq_length,0])
                y.append(dataset[i+seq_length,0])
            return np.array(X), np.array(y)
        
        seq_length=st.slider("Select the Sequence Length", 1, 30, 10, key="lstm_seq")
        
        # Check if we have enough data
        if len(train_data) <= seq_length or len(test_data) <= seq_length:
            st.error("Not enough data for the selected sequence length. Try reducing the sequence length.")
        else:
            train_X, train_y = create_sequences(train_data, seq_length)
            test_X, test_y = create_sequences(test_data, seq_length)
            
            # Reshape train_X and test_X
            train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
            test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))
            
            # Build LSTM Model 
            units = st.slider("Number of LSTM units", 10, 100, 50)
            epochs = st.slider("Number of epochs", 5, 50, 20)
            batch_size = st.slider("Batch size", 4, 32, 16)
            
            lstm_model=Sequential()
            lstm_model.add(LSTM(units=units, return_sequences=True, input_shape=(train_X.shape[1],1)))
            lstm_model.add(LSTM(units=units))
            lstm_model.add(Dense(units=1))
            
            # Compile the lstm_model 
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Add progress bar for training
            with st.spinner('Training LSTM model...'):
                lstm_model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Predict the future values
            train_predictions=lstm_model.predict(train_X)
            test_predictions=lstm_model.predict(test_X)
            
            # Inverse transform
            train_predictions=scaler.inverse_transform(train_predictions)
            test_predictions=scaler.inverse_transform(test_predictions)
            
            # Calculate the mean_squared_error
            actual_train = scaler.inverse_transform(train_data[seq_length:])
            actual_test = scaler.inverse_transform(test_data[seq_length:])
            
            train_mse=mean_squared_error(actual_train, train_predictions)
            train_rmse=np.sqrt(train_mse)
            test_mse=mean_squared_error(actual_test, test_predictions)
            test_rmse=np.sqrt(test_mse)
            
            st.write(f"Train Root Mean Squared Error: {train_rmse}")
            st.write(f"Test Root Mean Squared Error: {test_rmse}")
            
            # Prepare for plotting
            train_dates=data["Date"][seq_length:train_size]
            test_dates=data["Date"][train_size+seq_length:len(data)]
            
            # Future predictions
            if len(test_data) >= seq_length:
                # Use the last sequence from the data
                last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
                
                # Make future predictions one step at a time
                future_preds = []
                current_seq = last_sequence.copy()
                
                for _ in range(forecast_period):
                    # Predict next value
                    next_pred = lstm_model.predict(current_seq)
                    future_preds.append(next_pred[0, 0])
                    
                    # Update sequence for next prediction (remove oldest, add new)
                    current_seq = np.append(current_seq[:, 1:, :], 
                                          np.array([[[next_pred[0, 0]]]]), axis=1)
                
                # Convert predictions back to original scale
                future_preds = np.array(future_preds).reshape(-1, 1)
                future_preds = scaler.inverse_transform(future_preds)
                
                # Create future dates
                last_date = pd.to_datetime(data["Date"].iloc[-1])
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_period)
                
                future_df = pd.DataFrame({'Date': future_dates, 'Predicted': future_preds.flatten()})
                st.write("Future Predictions:", future_df)
                
                # Plot the data
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=test_dates, y=test_predictions.flatten(), mode='lines', name='Test Predictions', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Predicted"], mode='lines', name='Future Predictions', line=dict(color='red')))
                fig.update_layout(title='Actual vs Predicted (LSTM)', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
                st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error in LSTM model: {e}")
        st.info("Try adjusting the parameters or selecting a different column.")

elif selected_model=="Prophet":
    st.header("Facebook Prophet Model")
    
    try:
        prophet_data=data[["Date", column]].copy()
        prophet_data=prophet_data.rename(columns={"Date":"ds", column:"y"})
        
        # Convert ds to datetime if it's not already
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        
        # fit the prophet model
        prophet_model=Prophet(
            yearly_seasonality=st.checkbox("Yearly Seasonality", True),
            weekly_seasonality=st.checkbox("Weekly Seasonality", True),
            daily_seasonality=st.checkbox("Daily Seasonality", False),
            seasonality_mode=st.selectbox("Seasonality Mode", ["additive", "multiplicative"])
        )
        
        with st.spinner('Training Prophet model...'):
            prophet_model.fit(prophet_data)
        
        # Forecast the future values
        future = prophet_model.make_future_dataframe(periods=forecast_period)
        forecast = prophet_model.predict(future)
        
        # Display forecast dataframe
        st.write("Forecast Data:")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period))
        
        # Plot the forecast using plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prophet_data['ds'], y=prophet_data['y'], mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(color='rgba(255,0,0,0.3)')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(color='rgba(255,0,0,0.3)')))
        fig.update_layout(title='Forecast with Facebook Prophet', xaxis_title='Date', yaxis_title='Value', width=1000, height=500)
        st.plotly_chart(fig)
        
        # Plot the components using matplotlib
        fig2 = prophet_model.plot_components(forecast)
        st.pyplot(fig2)
        
        # Display performance metrics (for the historical fit)
        st.subheader("Performance Metrics")
        historical_forecast = forecast[forecast['ds'].isin(prophet_data['ds'])]
        mae = mean_absolute_error(prophet_data['y'], historical_forecast['yhat'])
        mse = mean_squared_error(prophet_data['y'], historical_forecast['yhat'])
        rmse = np.sqrt(mse)
        st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    except Exception as e:
        st.error(f"Error in Prophet model: {e}")
        st.info("Try selecting a different column or date range.")

elif selected_model == "GRU":
    st.header("Gated Recurrent Unit (GRU)")
    
    try:
        # Scale the Data
        scaler=MinMaxScaler(feature_range=(0,1))
        # Prepare the data for GRU
        scaled_data = scaler.fit_transform(data[column].values.reshape(-1, 1))
        train_size = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
        
        def create_sequences(dataset, seq_length):
            X, y = [], []
            for i in range(len(dataset) - seq_length):
                X.append(dataset[i:i + seq_length, 0])
                y.append(dataset[i + seq_length, 0])
            return np.array(X), np.array(y)
        
        seq_length = st.slider("Select the Sequence Length", 1, 30, 10, key="gru_seq")
        
        # Check if we have enough data
        if len(train_data) <= seq_length or len(test_data) <= seq_length:
            st.error("Not enough data for the selected sequence length. Try reducing the sequence length.")
        else:
            train_X, train_y = create_sequences(train_data, seq_length)
            test_X, test_y = create_sequences(test_data, seq_length)
            
            # Reshape train_X and test_X for GRU
            train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
            test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))
            
            # Build GRU model with customizable parameters
            units = st.slider("Number of GRU units", 10, 100, 50, key="gru_units")
            epochs = st.slider("Number of epochs", 5, 50, 20, key="gru_epochs")
            batch_size = st.slider("Batch size", 4, 32, 16, key="gru_batch")
            
            gru_model = Sequential()
            gru_model.add(GRU(units=units, return_sequences=True, input_shape=(train_X.shape[1], 1)))
            gru_model.add(GRU(units=units))
            gru_model.add(Dense(units=1))
            
            # Compile the GRU model
            gru_model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train with progress information
            with st.spinner('Training GRU model...'):
                gru_model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Predict the values
            train_predictions = gru_model.predict(train_X)
            test_predictions = gru_model.predict(test_X)
            
            # Inverse transform
            train_predictions = scaler.inverse_transform(train_predictions)
            test_predictions = scaler.inverse_transform(test_predictions)
            
            # Get the actual values for comparison
            actual_train = scaler.inverse_transform(train_data[seq_length:])
            actual_test = scaler.inverse_transform(test_data[seq_length:])
            
            # Calculate metrics
            train_mse = mean_squared_error(actual_train, train_predictions)
            train_rmse = np.sqrt(train_mse)
            test_mse = mean_squared_error(actual_test, test_predictions)
            test_rmse = np.sqrt(test_mse)
            
            st.write(f"Train Root Mean Squared Error: {train_rmse}")
            st.write(f"Test Root Mean Squared Error: {test_rmse}")
            
            # Prepare for plotting
            train_dates = data["Date"][seq_length:train_size]
            test_dates = data["Date"][train_size+seq_length:len(data)]
            
            # Future predictions
            if len(test_data) >= seq_length:
                # Use the last sequence from the data
                last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
                
                # Make future predictions one step at a time
                future_preds = []
                current_seq = last_sequence.copy()
                
                for _ in range(forecast_period):
                    # Predict next value
                    next_pred = gru_model.predict(current_seq)
                    future_preds.append(next_pred[0, 0])
                    
                    # Update sequence for next prediction (remove oldest, add new)
                    current_seq = np.append(current_seq[:, 1:, :], 
                                          np.array([[[next_pred[0, 0]]]]), axis=1)
                
                # Convert predictions back to original scale
                future_preds = np.array(future_preds).reshape(-1, 1)
                future_preds = scaler.inverse_transform(future_preds)
                
                # Create future dates
                last_date = pd.to_datetime(data["Date"].iloc[-1])
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_period)
                
                future_df = pd.DataFrame({'Date': future_dates, 'Predicted': future_preds.flatten()})
                st.write("Future Predictions:", future_df)
                
                # Plot the data
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=test_dates, y=test_predictions.flatten(), mode='lines', name='Test Predictions', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Predicted"], mode='lines', name='Future Predictions', line=dict(color='red')))
                fig.update_layout(title='Actual vs Predicted (GRU)', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
                st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error in GRU model: {e}")
        st.info("Try adjusting the parameters or selecting a different column.")

elif selected_model == "DenseNet":
    st.header("DenseNet Model")
    
    try:
        # Preprocessing Data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[column].values.reshape(-1, 1))
        train_size = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
        
        def create_sequences(dataset, seq_length):
            X, y = [], []
            for i in range(len(dataset) - seq_length):
                X.append(dataset[i:i + seq_length, 0])
                y.append(dataset[i + seq_length, 0])
            return np.array(X), np.array(y)
        
        seq_length = st.slider("Select the Sequence Length", 1, 30, 10, key="dense_seq")
        
        # Check if we have enough data
        if len(train_data) <= seq_length or len(test_data) <= seq_length:
            st.error("Not enough data for the selected sequence length. Try reducing the sequence length.")
        else:
            train_X, train_y = create_sequences(train_data, seq_length)
            test_X, test_y = create_sequences(test_data, seq_length)
            
            # Reshape for DenseNet (flatten input)
            train_X = train_X.reshape(train_X.shape[0], train_X.shape[1])
            test_X = test_X.reshape(test_X.shape[0], test_X.shape[1])
            
            # Build DenseNet model with customizable parameters
            units1 = st.slider("First layer units", 32, 256, 128, key="dense_units1")
            units2 = st.slider("Second layer units", 16, 128, 64, key="dense_units2")
            epochs = st.slider("Number of epochs", 5, 50, 20, key="dense_epochs")
            batch_size = st.slider("Batch size", 4, 32, 16, key="dense_batch")
            
            model = Sequential()
            model.add(Dense(units1, input_shape=(seq_length,), activation='relu'))
            model.add(Dense(units2, activation='relu'))
            model.add(Dense(1))
            
            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train the model
            with st.spinner('Training DenseNet model...'):
                model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Predict the values
            train_predictions = model.predict(train_X)
            test_predictions = model.predict(test_X)
            
            # Inverse transform
            train_predictions = scaler.inverse_transform(train_predictions)
            test_predictions = scaler.inverse_transform(test_predictions)
            
            # Get the actual values for comparison
            actual_train = scaler.inverse_transform(train_data[seq_length:])
            actual_test = scaler.inverse_transform(test_data[seq_length:])
            
            # Calculate metrics
            train_mse = mean_squared_error(actual_train, train_predictions)
            train_rmse = np.sqrt(train_mse)
            test_mse = mean_squared_error(actual_test, test_predictions)
            test_rmse = np.sqrt(test_mse)
            
            st.write(f"Train Root Mean Squared Error: {train_rmse}")
            st.write(f"Test Root Mean Squared Error: {test_rmse}")
            
            # Prepare for plotting
            train_dates = data["Date"][seq_length:train_size]
            test_dates = data["Date"][train_size+seq_length:len(data)]
            
            # Future predictions
            if len(test_data) >= seq_length:
                # Use the last sequence from the data
                last_sequence = scaled_data[-seq_length:].reshape(1, -1)
                
                # Make future predictions one step at a time
                future_preds = []
                current_seq = last_sequence.copy()
                
                for _ in range(forecast_period):
                    # Predict next value
                    next_pred = model.predict(current_seq)
                    future_preds.append(next_pred[0, 0])
                    
                    # Update sequence for next prediction (remove oldest, add new)
                    current_seq = np.append(current_seq[:, 1:], [[next_pred[0, 0]]], axis=1)
                
                # Convert predictions back to original scale
                future_preds = np.array(future_preds).reshape(-1, 1)
                future_preds = scaler.inverse_transform(future_preds)
                
                # Create future dates
                last_date = pd.to_datetime(data["Date"].iloc[-1])
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_period)
                
                future_df = pd.DataFrame({'Date': future_dates, 'Predicted': future_preds.flatten()})
                st.write("Future Predictions:", future_df)
                
                # Plot the data
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=test_dates, y=test_predictions.flatten(), mode='lines', name='Test Predictions', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Predicted"], mode='lines', name='Future Predictions', line=dict(color='red')))
                fig.update_layout(title='Actual vs Predicted (DenseNet)', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
                st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error in DenseNet model: {e}")
        st.info("Try adjusting the parameters or selecting a different column.")
else:
    st.write("Invalid model selected.")

st.write("Model selected:", selected_model)
st.sidebar.markdown("---")

# add author name and info in the sidebar
st.sidebar.markdown("### Author: Maria Nadeem🎉🎊⚡")
st.sidebar.markdown("### GitHub: [GitHub](https://github.com/marianadeem755)")
st.sidebar.markdown("### LinkedIn: [LinkedIn Account](https://www.linkedin.com/in/maria-nadeem-4994122aa/)")
st.sidebar.markdown("### Contact: [Email](mailto:marianadeem755@gmail.com)")
st.sidebar.markdown("### Credits: [codanics](https://codanics.com/)")
st.sidebar.markdown("---")

# URLs of the images
github_url = "https://img.icons8.com/fluent/48/000000/github.png"

# Adding a footer
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
    f'<div class="footer">Made by Maria Nadeem <a href="https://github.com/marianadeem755"><img src="{github_url}" width="30" height="30"></a> '
    f'<a href="https://codanics.com/">Credits: https://codanics.com/</a></div>',
    unsafe_allow_html=True
)
                    
