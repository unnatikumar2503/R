# Financial Market Analysis and Prediction 

# Install and load necessary packages
install.packages(c("tidyverse", "quantmod", "lubridate", "forecast", "ggplot2"))

library(tidyverse)
library(quantmod)
library(lubridate)
library(forecast)
library(ggplot2)

# Step 1: Setting Up the Environment
# Define the stock ticker and the date range
stock_ticker <- "AAPL" # Example: Apple Inc.
start_date <- as.Date("2010-01-01")
end_date <- Sys.Date()

# Get the stock data using quantmod package
getSymbols(stock_ticker, src = "yahoo", from = start_date, to = end_date)

# Check the structure of the data
str(get(stock_ticker))

# Convert the data to a data frame
stock_data <- data.frame(Date = index(get(stock_ticker)), coredata(get(stock_ticker)))

# View the first few rows
head(stock_data)

# Step 2: Exploratory Data Analysis (EDA)
# Visualize the closing prices over time
ggplot(stock_data, aes(x = Date, y = AAPL.Close)) +
  geom_line(color = "blue") +
  labs(title = paste("Closing Prices of", stock_ticker),
       x = "Date",
       y = "Closing Price") +
  theme_minimal()

# Calculate moving averages
stock_data <- stock_data %>%
  mutate(SMA_50 = rollmean(AAPL.Close, 50, fill = NA),
         SMA_200 = rollmean(AAPL.Close, 200, fill = NA))

# Plot closing prices with moving averages
ggplot(stock_data, aes(x = Date)) +
  geom_line(aes(y = AAPL.Close), color = "blue", alpha = 0.6) +
  geom_line(aes(y = SMA_50), color = "red") +
  geom_line(aes(y = SMA_200), color = "green") +
  labs(title = paste("Closing Prices with Moving Averages of", stock_ticker),
       x = "Date",
       y = "Price") +
  theme_minimal()

# Step 3: Time Series Analysis
# Convert to time series object
ts_data <- ts(stock_data$AAPL.Close, start = c(year(start_date), month(start_date)), frequency = 365)

# Decompose the time series
decomposed <- decompose(ts_data)
plot(decomposed)

# Step 4: Building a Predictive Model
# Fit ARIMA model
fit <- auto.arima(ts_data)

# Print model summary
summary(fit)

# Step 5: Forecast future prices
# Forecast the next 30 days
forecasted <- forecast(fit, h = 30)

# Plot the forecast
autoplot(forecasted) +
  labs(title = paste("30-Day Forecast for", stock_ticker),
       x = "Date",
       y = "Price") +
  theme_minimal()

# Step 6: Evaluating the Model
# Check residuals
checkresiduals(fit)

# Calculate accuracy
# Split the data into training and test sets
train_size <- floor(0.8 * nrow(stock_data))
train_data <- head(stock_data, train_size)
test_data <- tail(stock_data, nrow(stock_data) - train_size)

# Convert training data to time series
train_ts <- ts(train_data$AAPL.Close, start = c(year(start_date), month(start_date)), frequency = 365)

# Fit ARIMA model on training data
fit_train <- auto.arima(train_ts)

# Forecast on test data
forecasted_test <- forecast(fit_train, h = nrow(test_data))

# Calculate accuracy
accuracy(forecasted_test, test_data$AAPL.Close)

# Step 7: Conclusion and Documentation
# Document the findings and insights gained from the analysis and modeling
# Write a brief report summarizing the key steps, results, and conclusions
cat("Summary Report:
1. Collected and visualized historical stock prices for", stock_ticker, ".
2. Performed EDA including visualization of closing prices and moving averages.
3. Decomposed the time series to understand its components.
4. Built an ARIMA model for predicting future prices.
5. Evaluated the model's performance using residual analysis and accuracy metrics.
6. Forecasted the next 30 days of stock prices.
")
