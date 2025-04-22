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
from datetime import date
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Set page config
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title("Stock Market Price Prediction App")
st.text("This App Forecasts The Stock Market Price of Selected Company")
st.image("https://images.unsplash.com/photo-1563986768711-b3bde3dc821e?q=80&w=1468&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")

# Sidebar parameters
st.sidebar.header("Select the parameters")
start_date = st.sidebar.date_input('Start Date', date(2001, 1, 1))
end_date = st.sidebar.date_input('End Date', date(2025, 1, 1))  # Extended to 2025
ticker_list = ["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker = st.sidebar.selectbox("Select the Company", ticker_list)

# Fetch the data using yfinance library
@st.cache_data  # Cache the data to avoid repeated downloads
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data.rename(columns={'index': 'Date'}, inplace=True)
    return data

try:
    data = load_data(ticker, start_date, end_date)
    if data.empty:
        st.error("No data available for the selected date range. Please try different dates.")
    else:
        st.write('Data From', start_date, 'to', end_date)
        st.write(data)

        # Plot the Data 
        st.header("Data Visualization")
        st.subheader("Closing Price Over Time")
        
        # Fix: Use the correct column name for y-axis
        fig = px.line(data, x="Date", y="Close", title=f"{ticker} Closing Price", width=1000, height=600)
        st.plotly_chart(fig)
        
        # Create a selection box to choose the column for forecasting
        column_options = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        column = st.selectbox("Select the column for forecasting", column_options)
        
        # Prepare data for forecasting
        forecast_data = data[["Date", column]].copy()
        st.write("Selected Data for Forecasting")
        st.write(forecast_data)
        
        # Apply the ADF test to check the stationarity
        st.write("#### ADF Test Results")
        adf_result = adfuller(forecast_data[column].dropna())
        st.write(f"ADF Statistic: {adf_result[0]}")
        st.write(f"p-value: {adf_result[1]}")
        st.write(f"Is the data stationary? {adf_result[1] < 0.05}")
        
        # Decompose the Data if enough data points are available
        if len(forecast_data) > 2 * 12:  # Need at least 2 periods for decomposition
            try:
                st.write("#### Decomposition Analysis")
                decomposition = seasonal_decompose(forecast_data.set_index('Date')[column], model='additive', period=12)
                
                # Plot decomposition using Plotly
                st.write("Trend Component")
                fig_trend = px.line(x=forecast_data['Date'], y=decomposition.trend, 
                                  title="Trend Component", width=1000, height=400)
                st.plotly_chart(fig_trend)
                
                st.write("Seasonal Component")
                fig_seasonal = px.line(x=forecast_data['Date'], y=decomposition.seasonal, 
                                     title="Seasonal Component", width=1000, height=400)
                st.plotly_chart(fig_seasonal)
                
                st.write("Residual Component")
                fig_resid = px.line(x=forecast_data['Date'], y=decomposition.resid, 
                                  title="Residual Component", width=1000, height=400)
                st.plotly_chart(fig_resid)
            except Exception as e:
                st.warning(f"Could not perform decomposition: {str(e)}")
        
        # Select the Model
        models = ["SARIMA", "Random Forest", "LSTM", "Prophet", "GRU", "SVM", "DenseNet"]
        selected_model = st.sidebar.selectbox("Select the Model for Forecasting", models)
        
        if selected_model == "SARIMA":
            st.header("SARIMA Model")
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.slider("Select the value of p", 0, 5, 1)
            with col2:
                d = st.slider("Select the value of d", 0, 2, 1)
            with col3:
                q = st.slider("Select the value of q", 0, 5, 1)
            
            seasonal_order = st.slider("Select the seasonal period", 4, 24, 12)
            
            try:
                model = sm.tsa.statespace.SARIMAX(
                    forecast_data[column],
                    order=(p, d, q),
                    seasonal_order=(p, d, q, seasonal_order),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                model_fit = model.fit(disp=False)
                
                st.header("Model Summary")
                st.write(model_fit.summary())
                
                # Forecasting
                forecast_period = st.number_input("Select the number of days to forecast", 1, 365, 30)
                forecast = model_fit.get_forecast(steps=forecast_period)
                forecast_mean = forecast.predicted_mean
                conf_int = forecast.conf_int()
                
                # Create forecast dataframe
                last_date = forecast_data['Date'].iloc[-1]
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_period)
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Predicted': forecast_mean.values,
                    'Lower CI': conf_int.iloc[:, 0].values,
                    'Upper CI': conf_int.iloc[:, 1].values
                })
                
                st.write("Forecast Results")
                st.write(forecast_df)
                
                # Plot results
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=forecast_data['Date'],
                    y=forecast_data[column],
                    name='Actual',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Predicted'],
                    name='Forecast',
                    line=dict(color='red')
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Upper CI'],
                    fill=None,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Lower CI'],
                    fill='tonexty',
                    mode='lines',
                    line=dict(width=0),
                    name='Confidence Interval'
                ))
                fig.update_layout(
                    title=f"{ticker} {column} Price Forecast with SARIMA",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    width=1000,
                    height=600
                )
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error in SARIMA model: {str(e)}")
                
        elif selected_model == "Random Forest":
            st.header("Random Forest Regression")
            
            # Prepare features (using time-based features)
            forecast_data['Days'] = (forecast_data['Date'] - forecast_data['Date'].min()).dt.days
            X = forecast_data[['Days']].values
            y = forecast_data[column].values
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Train model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = rf_model.predict(X_train)
            y_pred_test = rf_model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            st.write(f"Train RMSE: {train_rmse:.4f}")
            st.write(f"Test RMSE: {test_rmse:.4f}")
            
            # Create full prediction series for plotting
            full_pred = rf_model.predict(X)
            
            # Plot results
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=y,
                name='Actual',
                line=dict(color='blue')
            )
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=full_pred,
                name='Predicted',
                line=dict(color='red')
            )
            fig.add_vline(
                x=forecast_data['Date'].iloc[train_size],
                line_dash="dash",
                line_color="green",
                annotation_text="Train/Test Split"
            )
            fig.update_layout(
                title=f"{ticker} {column} Price - Random Forest",
                xaxis_title="Date",
                yaxis_title="Price",
                width=1000,
                height=600
            )
            st.plotly_chart(fig)
            
        elif selected_model == "LSTM":
            st.header("Long Short-Term Memory (LSTM)")
            
            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(forecast_data[[column]])
            
            # Create sequences
            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i:i + seq_length, 0])
                    y.append(data[i + seq_length, 0])
                return np.array(X), np.array(y)
            
            seq_length = st.slider("Sequence Length", 5, 60, 30)
            X, y = create_sequences(scaled_data, seq_length)
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Reshape for LSTM [samples, time steps, features]
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Build model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=1
            )
            
            # Plot training history
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=history.history['loss'],
                name='Train Loss',
                mode='lines'
            ))
            fig_loss.add_trace(go.Scatter(
                y=history.history['val_loss'],
                name='Validation Loss',
                mode='lines'
            ))
            fig_loss.update_layout(
                title='Model Training History',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                width=800,
                height=400
            )
            st.plotly_chart(fig_loss)
            
            # Make predictions
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)
            
            # Inverse transform
            train_predict = scaler.inverse_transform(train_predict)
            y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
            test_predict = scaler.inverse_transform(test_predict)
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
            test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
            st.write(f"Train RMSE: {train_rmse:.4f}")
            st.write(f"Test RMSE: {test_rmse:.4f}")
            
            # Create full prediction series for plotting
            train_predict_plot = np.empty_like(scaled_data)
            train_predict_plot[:, :] = np.nan
            train_predict_plot[seq_length:seq_length + len(train_predict), :] = train_predict
            
            test_predict_plot = np.empty_like(scaled_data)
            test_predict_plot[:, :] = np.nan
            test_predict_plot[seq_length + len(train_predict):, :] = test_predict
            
            # Plot results
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data[column],
                name='Actual',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=scaler.inverse_transform(train_predict_plot).flatten(),
                name='Train Predictions',
                line=dict(color='green')
            ))
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=scaler.inverse_transform(test_predict_plot).flatten(),
                name='Test Predictions',
                line=dict(color='red')
            ))
            fig.add_vline(
                x=forecast_data['Date'].iloc[train_size + seq_length],
                line_dash="dash",
                line_color="purple",
                annotation_text="Train/Test Split"
            )
            fig.update_layout(
                title=f"{ticker} {column} Price - LSTM Predictions",
                xaxis_title="Date",
                yaxis_title="Price",
                width=1000,
                height=600
            )
            st.plotly_chart(fig)
            
        elif selected_model == "Prophet":
            st.header("Facebook Prophet Model")
            
            # Prepare data for Prophet
            prophet_data = forecast_data[['Date', column]].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Create and fit model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='additive'
            )
            model.fit(prophet_data)
            
            # Make future dataframe
            future_periods = st.slider("Forecast Period (days)", 30, 365, 90)
            future = model.make_future_dataframe(periods=future_periods)
            
            # Make forecast
            forecast = model.predict(future)
            
            # Plot forecast
            fig1 = model.plot(forecast)
            plt.title(f"{ticker} {column} Price Forecast with Prophet")
            st.pyplot(fig1)
            
            # Plot components
            st.write("Forecast Components")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)
            
            # Show forecast dataframe
            st.write("Forecast Data")
            st.write(forecast.tail()[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
            
        elif selected_model == "GRU":
            st.header("Gated Recurrent Unit (GRU)")
            
            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(forecast_data[[column]])
            
            # Create sequences
            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i:i + seq_length, 0])
                    y.append(data[i + seq_length, 0])
                return np.array(X), np.array(y)
            
            seq_length = st.slider("Sequence Length", 5, 60, 30)
            X, y = create_sequences(scaled_data, seq_length)
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Reshape for GRU [samples, time steps, features]
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Build model
            model = Sequential()
            model.add(GRU(50, return_sequences=True, input_shape=(seq_length, 1)))
            model.add(GRU(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=1
            )
            
            # Plot training history
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=history.history['loss'],
                name='Train Loss',
                mode='lines'
            ))
            fig_loss.add_trace(go.Scatter(
                y=history.history['val_loss'],
                name='Validation Loss',
                mode='lines'
            ))
            fig_loss.update_layout(
                title='Model Training History',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                width=800,
                height=400
            )
            st.plotly_chart(fig_loss)
            
            # Make predictions
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)
            
            # Inverse transform
            train_predict = scaler.inverse_transform(train_predict)
            y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
            test_predict = scaler.inverse_transform(test_predict)
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
            test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
            st.write(f"Train RMSE: {train_rmse:.4f}")
            st.write(f"Test RMSE: {test_rmse:.4f}")
            
            # Create full prediction series for plotting
            train_predict_plot = np.empty_like(scaled_data)
            train_predict_plot[:, :] = np.nan
            train_predict_plot[seq_length:seq_length + len(train_predict), :] = train_predict
            
            test_predict_plot = np.empty_like(scaled_data)
            test_predict_plot[:, :] = np.nan
            test_predict_plot[seq_length + len(train_predict):, :] = test_predict
            
            # Plot results
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data[column],
                name='Actual',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=scaler.inverse_transform(train_predict_plot).flatten(),
                name='Train Predictions',
                line=dict(color='green')
            ))
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=scaler.inverse_transform(test_predict_plot).flatten(),
                name='Test Predictions',
                line=dict(color='red')
            ))
            fig.add_vline(
                x=forecast_data['Date'].iloc[train_size + seq_length],
                line_dash="dash",
                line_color="purple",
                annotation_text="Train/Test Split"
            )
            fig.update_layout(
                title=f"{ticker} {column} Price - GRU Predictions",
                xaxis_title="Date",
                yaxis_title="Price",
                width=1000,
                height=600
            )
            st.plotly_chart(fig)
            
        elif selected_model == "SVM":
            st.header("Support Vector Machine (SVM)")
            
            # Prepare features (using time-based features)
            forecast_data['Days'] = (forecast_data['Date'] - forecast_data['Date'].min()).dt.days
            X = forecast_data[['Days']].values
            y = forecast_data[column].values
            
            # Scale data
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            y_scaled = scaler.fit_transform(y.reshape(-1, 1))
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
            y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
            
            # Train model
            svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
            svm_model.fit(X_train, y_train.ravel())
            
            # Make predictions
            y_pred_train = svm_model.predict(X_train)
            y_pred_test = svm_model.predict(X_test)
            
            # Inverse transform
            y_pred_train = scaler.inverse_transform(y_pred_train.reshape(-1, 1))
            y_train_actual = scaler.inverse_transform(y_train)
            y_pred_test = scaler.inverse_transform(y_pred_test.reshape(-1, 1))
            y_test_actual = scaler.inverse_transform(y_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_test))
            st.write(f"Train RMSE: {train_rmse:.4f}")
            st.write(f"Test RMSE: {test_rmse:.4f}")
            
            # Create full prediction series for plotting
            full_pred_scaled = svm_model.predict(X_scaled)
            full_pred = scaler.inverse_transform(full_pred_scaled.reshape(-1, 1))
            
            # Plot results
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=y,
                name='Actual',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=full_pred.flatten(),
                name='Predicted',
                line=dict(color='red')
            ))
            fig.add_vline(
                x=forecast_data['Date'].iloc[train_size],
                line_dash="dash",
                line_color="green",
                annotation_text="Train/Test Split"
            )
            fig.update_layout(
                title=f"{ticker} {column} Price - SVM Predictions",
                xaxis_title="Date",
                yaxis_title="Price",
                width=1000,
                height=600
            )
            st.plotly_chart(fig)
            
        elif selected_model == "DenseNet":
            st.header("DenseNet Model")
            
            # Prepare features (using time-based features)
            forecast_data['Days'] = (forecast_data['Date'] - forecast_data['Date'].min()).dt.days
            X = forecast_data[['Days']].values
            y = forecast_data[column].values
            
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
            model.add(Dense(64, activation='relu', input_shape=(1,)))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=1
            )
            
            # Plot training history
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=history.history['loss'],
                name='Train Loss',
                mode='lines'
            ))
            fig_loss.add_trace(go.Scatter(
                y=history.history['val_loss'],
                name='Validation Loss',
                mode='lines'
            ))
            fig_loss.update_layout(
                title='Model Training History',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                width=800,
                height=400
            )
            st.plotly_chart(fig_loss)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Inverse transform
            y_pred_train = scaler.inverse_transform(y_pred_train)
            y_train_actual = scaler.inverse_transform(y_train)
            y_pred_test = scaler.inverse_transform(y_pred_test)
            y_test_actual = scaler.inverse_transform(y_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_test))
            st.write(f"Train RMSE: {train_rmse:.4f}")
            st.write(f"Test RMSE: {test_rmse:.4f}")
            
            # Create full prediction series for plotting
            full_pred_scaled = model.predict(X_scaled)
            full_pred = scaler.inverse_transform(full_pred_scaled)
            
            # Plot results
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=y,
                name='Actual',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=full_pred.flatten(),
                name='Predicted',
                line=dict(color='red')
            ))
            fig.add_vline(
                x=forecast_data['Date'].iloc[train_size],
                line_dash="dash",
                line_color="green",
                annotation_text="Train/Test Split"
            )
            fig.update_layout(
                title=f"{ticker} {column} Price - DenseNet Predictions",
                xaxis_title="Date",
                yaxis_title="Price",
                width=1000,
                height=600
            )
            st.plotly_chart(fig)
            
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Author: Maria Nadeem")
st.sidebar.markdown("### GitHub: [GitHub Profile](https://github.com/marianadeem755)")
st.sidebar.markdown("### LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/maria-nadeem-4994122aa/)")
st.sidebar.markdown("### Contact: [Email](mailto:marianadeem755@gmail.com)")
st.sidebar.markdown("---")

# Main footer
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
    '<div class="footer">'
    'Made by Maria Nadeem | '
    '<a href="https://github.com/marianadeem755" target="_blank">GitHub</a> | '
    '<a href="https://codanics.com/" target="_blank">Credits: Codanics</a>'
    '</div>',
    unsafe_allow_html=True
)
