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
st.write("select the *specific date* from the date range or zoom in the plot for detailed visualization and select the specific column")
fig=px.line(data, x="Date", y=data.index, title="Closing price of the stock", width=1000, height=600)
st.plotly_chart(fig)
# create a selection box to choose the column for forecasting
column=st.selectbox("Select the column for forecasting", data.columns[1:])
data=data[["Date", column]]
st.write("Selected Data for forecasting")
st.write(data)
# Apply the ADF test to check the stationarity
st.write("#### ADF test to check the stationarity")
st.write(adfuller(data[column])[1]<0.05)
# Decompose the Data and make the decomposition plot 
decomposition=seasonal_decompose(data[column], model='additive', period=12)
st.write(decomposition.plot())
# Now making the decomposition plot using plotly
st.write("Decomposition Plot using plotly")
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, width=1000, height=400, title="Trend", labels={"x":"Date", "y":"price"}).update_traces(line_color="green"))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, width=1000, height=400, title="Seasonality", labels={"x":"Date", "y":"price"}).update_traces(line_color="blue"))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, width=1000, height=400, title="Residual", labels={"x":"Date", "y":"price"}).update_traces(line_color="red"))
# Select the Model
models=["SARIMA","Random Forest","LSTM", "Prophet","GRU","SVM","DenseNet"]
selected_model=st.sidebar.selectbox("Select the Model for Forecasting", models)
if selected_model=="SARIMA":
    p=st.slider("Select the value of p", 0,5,2)
    d=st.slider("Selct the value of d", 0,5,1)
    q=st.slider("Select the Value of q", 0,5,2)
    seasonal_order=st.number_input("Select the value of p", 0,24,12)
    model=sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q), seasonal_order=(p,d,q, seasonal_order))
    # Train the model 
    model=model.fit()
    #Summary of the model 
    st.header("Model Summary")
    st.write(model.summary())
    st.write("---")
     # Forecasting using SARIMA
    st.write("<p style='color:red; font-size: 50px; font-weight: bold;'>Forecasting the data with SARIMA</p>",
             unsafe_allow_html=True)
    
    forecast_period=st.number_input("Select the Number of days to forecast", 1,365,10)
    predictions=model.get_prediction(start=len(data), end=len(data)+forecast_period)
    predictions=predictions.predicted_mean
    predictions.index=pd.date_range(start=end_date, periods=len(predictions), freq="D")
    predictions=pd.DataFrame(predictions)
    predictions.insert(0, "Date", predictions.index, True)
    predictions.reset_index(drop=True, inplace=True)
    st.write('predictions', predictions)
    st.write("Actual Data", data)
    st.write("---")
    # make the plotly plot
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data[column], name="Actual", mode="lines", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], name="Predicted", mode="lines", line=dict(color='red')))
    fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="predicted", width=1000, height=400)
    st.plotly_chart(fig)
elif selected_model=="Random Forest":
    st.header("Random Forest Regression")
    # Splitting Data into training and Testing set
    train_size=int(len(data)*0.8)
    train_data, test_data=data[:train_size], data[train_size:]
    # Feature Engineering 
    train_X, train_y=train_data["Date"], train_data[column]
    test_X,test_y=test_data["Date"], test_data[column]
    # train the random forest model 
    rf_model=RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(train_X.values.reshape(-1,1), train_y.values)
    # predictions
    predictions=rf_model.predict(test_X.values.reshape(-1,1))
    # calculate the mean_squared_error
    mse=mean_squared_error(test_y, predictions)
    rmse=np.sqrt(mse)
    st.write(f"Root Mean Squared Error: {rmse}")
    # combine the training and Testing Data for plotting
    combined_data=pd.concat([train_data, test_data])
    # plot the data 
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=combined_data["Date"], y=combined_data[column], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_data["Date"], y=predictions, mode='lines', name='predicted', line=dict(color='red')))
    fig.update_layout(title="Actual vs Predicted(R andom Forest)", xaxis_title="Date", yaxis_title="Price", width=1000, height=400)
    st.plotly_chart(fig)
elif selected_model == "SVM":
    st.header("Support Vector Machine (SVM)")
    # scale the Data
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
    seq_length = st.slider("Select the Sequence Length", 1, 30, 10)
    train_X, train_y = create_sequences(train_data, seq_length)
    test_X, test_y = create_sequences(test_data, seq_length)
    # Reshape train_X and test_X for SVM
    train_X = train_X.reshape(-1, seq_length)
    test_X = test_X.reshape(-1, seq_length)
    # Build and train SVM model
    svm_model = SVR(kernel='rbf')
    svm_model.fit(train_X, train_y)
    # Predict the future values
    train_predictions = svm_model.predict(train_X)
    test_predictions = svm_model.predict(test_X)
    train_predictions = train_predictions.reshape(-1, 1)
    test_predictions = test_predictions.reshape(-1, 1)
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    # Calculate the mean_squared_error
    train_mse = mean_squared_error(train_data[seq_length:], train_predictions)
    train_rmse = np.sqrt(train_mse)
    test_mse = mean_squared_error(test_data[seq_length:], test_predictions)
    test_rmse = np.sqrt(test_mse)
    st.write(f"Train Root Mean Squared Error: {train_rmse}")
    st.write(f"Test Root Mean Squared Error: {test_rmse}")
    train_dates = data["Date"][:train_size + seq_length]
    test_dates = data["Date"][train_size + seq_length:]
    combined_dates = pd.concat([train_dates, test_dates])
    combined_predictions = np.concatenate([train_predictions, test_predictions])
    # Plot the data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined_dates, y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_dates, y=combined_predictions, mode='lines', name='Predicted', line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted (SVM)', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
    st.plotly_chart(fig)
elif selected_model=="LSTM":
    st.header("Long Short Term Memory(LSTM)")
    # scale the Data
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(data[column].values.reshape(-1,1))
    #split the Data into Training and Testing Data 
    train_size=int(len(scaled_data)*0.8)
    train_data, test_data=scaled_data[:train_size], scaled_data[train_size:]
    def create_sequences(dataset, seq_length):
        X, y=[],[]
        for i in range(len(dataset)-seq_length):
            X.append(dataset[i:i+seq_length,0])
            y.append(dataset[i+seq_length,0])
        return np.array(X), np.array(y)
    seq_length=st.slider("Select the Sequence Length", 1,30,10)
    train_X, train_y = create_sequences(train_data, seq_length)
    test_X, test_y = create_sequences(test_data, seq_length)
    # Reshape train_X and test_X
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))
    # Build LSTM Model 
    lstm_model=Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(train_X.shape[1],1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(units=1))
    #compile the lstm_model 
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(train_X,train_y, epochs=20, batch_size=16)
    # predict the future values
    train_predictions=lstm_model.predict(train_X)
    test_predictions=lstm_model.predict(test_X)
    train_predictions=scaler.inverse_transform(train_predictions)
    test_predictions=scaler.inverse_transform(test_predictions)
    # calculate the mean_squared_error
    train_mse=mean_squared_error(train_data[seq_length:], train_predictions)
    train_rmse=np.sqrt(train_mse)
    test_mse=mean_squared_error(test_data[seq_length:], test_predictions)
    test_rmse=np.sqrt(test_mse)
    st.write(f"train Root Mean Squared Error:{train_rmse}")
    st.write(f"Test Root Mean Squared Error:{test_rmse}")
    train_dates=data["Date"][:train_size+seq_length]
    test_dates=data["Date"][train_size+seq_length:]
    combined_dates=pd.concat([train_dates,test_dates])
    combined_predictions=np.concatenate([train_predictions,test_predictions])
    # Plot the data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined_dates, y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_dates, y=combined_predictions, mode='lines', name='Predicted',
                             line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted (LSTM)', xaxis_title='Date', yaxis_title='Price',
                      width=1000, height=400)
    st.plotly_chart(fig)
elif selected_model=="Prophet":
    st.header("Facebook Prophet Model")
    prophet_data=data[["Date", column]]
    prophet_data=prophet_data.rename(columns={"Date":"ds", column:"y"})
    # fit the prophet model
    prophet_model=Prophet()
    prophet_model.fit(prophet_data)
        # Forecast the future values
    future = prophet_model.make_future_dataframe(periods=365)
    forecast = prophet_model.predict(future)

    # Plot the forecast
    fig = prophet_model.plot(forecast)
    plt.title('Forecast with Facebook Prophet')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig)
    # Plot the components (trend, seasonal, etc.)
    st.write("Plotting the components of the forecast")
    fig2 = prophet_model.plot_components(forecast)
    st.pyplot(fig2)

    # Display performance metrics (optional)
    st.write("Performance Metrics")
    actual_values = data[column][-forecast_period:].values
    predicted_values = forecast["yhat"][-forecast_period:].values
    mae = mean_absolute_error(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
elif selected_model == "GRU":
    st.header("Gated Recurrent Unit (GRU)")
     # scale the Data
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
    seq_length = st.slider("Select the Sequence Length", 1, 30, 10)
    train_X, train_y = create_sequences(train_data, seq_length)
    test_X, test_y = create_sequences(test_data, seq_length)
    # Reshape train_X and test_X for GRU
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))
    # Build GRU model
    gru_model = Sequential()
    gru_model.add(GRU(units=50, return_sequences=True, input_shape=(train_X.shape[1], 1)))
    gru_model.add(GRU(units=50))
    gru_model.add(Dense(units=1))
    # Compile the GRU model
    gru_model.compile(optimizer='adam', loss='mean_squared_error')
    gru_model.fit(train_X, train_y, epochs=20, batch_size=16)
    # Predict the future values
    train_predictions = gru_model.predict(train_X)
    test_predictions = gru_model.predict(test_X)
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    # Calculate the mean_squared_error
    train_mse = mean_squared_error(train_data[seq_length:], train_predictions)
    train_rmse = np.sqrt(train_mse)
    test_mse = mean_squared_error(test_data[seq_length:], test_predictions)
    test_rmse = np.sqrt(test_mse)
    st.write(f"Train Root Mean Squared Error: {train_rmse}")
    st.write(f"Test Root Mean Squared Error: {test_rmse}")
    train_dates = data["Date"][:train_size + seq_length]
    test_dates = data["Date"][train_size + seq_length:]
    combined_dates = pd.concat([train_dates, test_dates])
    combined_predictions = np.concatenate([train_predictions, test_predictions])
    # Plot the data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined_dates, y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_dates, y=combined_predictions, mode='lines', name='Predicted', line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted (GRU)', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
    st.plotly_chart(fig)
elif selected_model == "DenseNet":
    st.header("DenseNet Model")
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
    seq_length = st.slider("Select the Sequence Length", 1, 30, 10)
    train_X, train_y = create_sequences(train_data, seq_length)
    test_X, test_y = create_sequences(test_data, seq_length)
    # Reshape train_X and test_X for DenseNet
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1]))  # Remove the extra dimension
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1]))  # Remove the extra dimension
    # Build DenseNet model
    model = Sequential()
    model.add(Dense(128, input_shape=(seq_length,), activation='relu'))  # Adjust input_shape
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(train_X, train_y, epochs=20, batch_size=16)
    # Predict the future values
    train_predictions = model.predict(train_X)
    test_predictions = model.predict(test_X)
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    # Calculate the mean_squared_error
    train_mse = mean_squared_error(train_data[seq_length:], train_predictions)
    train_rmse = np.sqrt(train_mse)
    test_mse = mean_squared_error(test_data[seq_length:], test_predictions)
    test_rmse = np.sqrt(test_mse)
    st.write(f"Train Root Mean Squared Error: {train_rmse}")
    st.write(f"Test Root Mean Squared Error: {test_rmse}")
    train_dates = data["Date"][:train_size + seq_length]
    test_dates = data["Date"][train_size + seq_length:]
    combined_dates = pd.concat([train_dates, test_dates])
    combined_predictions = np.concatenate([train_predictions, test_predictions])
    # Plot the data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined_dates, y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_dates, y=combined_predictions, mode='lines', name='Predicted', line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted (DenseNet)', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
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
             f'<a href="https://codanics.com/">Credits: https://codanics.com/</a></div>',unsafe_allow_html=True
    )
