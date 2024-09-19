Link: https://stock-predictor-ambcnhqkzasbd5kfmi8qpd.streamlit.app/ 

This project aims to predict stock prices using historical data from 2010 to 2019. By leveraging machine learning and data visualization, the application allows users to input a stock ticker symbol and see both historical and predicted future trends.

Key components of the project:

Data Retrieval: Historical stock price data is fetched using the yfinance library, allowing for seamless access to stock data for the selected stock ticker. The data covers the period from January 2010 to December 2019.

Data Visualization: The project visualizes the closing prices over time, along with 100-day and 200-day moving averages (MA), providing users with insights into long-term trends.

Modeling with LSTM: A pre-trained Long Short-Term Memory (LSTM) model, loaded using Keras, predicts future stock prices based on the last 100 days of data. The dataset is split into training and testing sets, where the model predicts the closing prices for the testing period.

Interactive Dashboard: The app, built using Streamlit, enables users to enter any stock ticker symbol, view summary statistics, and visualize both the original and predicted prices side-by-side on a time-series plot.

Scaling and Inverse Scaling: The data is scaled using MinMaxScaler to normalize the values before feeding it into the LSTM model. After predictions, inverse scaling is applied to return the prices to their original scale.

This project provides an interactive and user-friendly platform to explore stock price trends and future predictions, utilizing deep learning and data visualization techniques.
