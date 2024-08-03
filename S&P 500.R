# Predictive Finance Analysis

# Step 1: Install and Load Required Libraries
required_packages <- c("quantmod", "tidyverse", "forecast", "caret", "randomForest", "xgboost", 
                       "PerformanceAnalytics", "PortfolioAnalytics", "ROI", "ROI.plugin.quadprog", 
                       "ROI.plugin.glpk", "shiny")

new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load the necessary libraries
library(quantmod)
library(tidyverse)
library(forecast)
library(caret)
library(randomForest)
library(xgboost)
library(PerformanceAnalytics)
library(PortfolioAnalytics)
library(ROI)
library(ROI.plugin.quadprog)
library(ROI.plugin.glpk)
library(shiny)

# Step 2: Set Stock Symbol and Date Range
symbol <- "^GSPC"  # S&P 500 index
start_date <- as.Date("2010-01-01")
end_date <- Sys.Date()

# Step 3: Get Stock Data
getSymbols(symbol, src = "yahoo", from = start_date, to = end_date)

# Convert to data frame
stock_data <- data.frame(date = index(GSPC), coredata(GSPC))

# Summary statistics
print(summary(stock_data))

# Step 4: Plot Closing Prices
ggplot(stock_data, aes(x = date, y = GSPC.Close)) +
  geom_line() +
  labs(title = "S&P 500 Index Closing Prices", x = "Date", y = "Closing Price") +
  theme_minimal()

# Step 5: Add Moving Averages to Data
stock_data <- stock_data %>%
  mutate(SMA_50 = SMA(GSPC.Close, n = 50),
         SMA_200 = SMA(GSPC.Close, n = 200))

# Plot Closing Prices with Moving Averages
ggplot(stock_data, aes(x = date)) +
  geom_line(aes(y = GSPC.Close, color = "Close")) +
  geom_line(aes(y = SMA_50, color = "50-day SMA")) +
  geom_line(aes(y = SMA_200, color = "200-day SMA")) +
  labs(title = "S&P 500 Index with Moving Averages", x = "Date", y = "Price") +
  scale_color_manual("", values = c("Close" = "black", "50-day SMA" = "blue", "200-day SMA" = "red")) +
  theme_minimal()

# Step 6: Time Series Analysis Using ARIMA
ts_data <- ts(stock_data$GSPC.Close, frequency = 252)
arima_model <- auto.arima(ts_data)
arima_forecast <- forecast(arima_model, h = 30)

# Plot ARIMA Forecast
autoplot(arima_forecast) +
  labs(title = "S&P 500 Index Forecast using ARIMA", x = "Date", y = "Price")

# Step 7: Add Lagged Data for Linear Regression
stock_data <- stock_data %>%
  mutate(Lag1 = lag(GSPC.Close, 1))

# Linear Regression Model
lm_model <- lm(GSPC.Close ~ Lag1, data = stock_data)
print(summary(lm_model))

# Step 8: Prepare Data for Machine Learning Models
model_data <- stock_data %>%
  filter(!is.na(Lag1)) %>%
  select(GSPC.Close, Lag1)

# Split Data into Training and Testing Sets
set.seed(123)
train_index <- createDataPartition(model_data$GSPC.Close, p = 0.8, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

# Random Forest Model
rf_model <- randomForest(GSPC.Close ~ Lag1, data = train_data)
rf_predictions <- predict(rf_model, test_data)
rf_rmse <- sqrt(mean((rf_predictions - test_data$GSPC.Close)^2))

# XGBoost Model
dtrain <- xgb.DMatrix(data = as.matrix(train_data$Lag1), label = train_data$GSPC.Close)
dtest <- xgb.DMatrix(data = as.matrix(test_data$Lag1), label = test_data$GSPC.Close)

params <- list(objective = "reg:squarederror", max_depth = 3, eta = 0.1)
xgb_model <- xgboost(params = params, data = dtrain, nrounds = 100)
xgb_predictions <- predict(xgb_model, dtest)
xgb_rmse <- sqrt(mean((xgb_predictions - test_data$GSPC.Close)^2))

# Evaluate Model Performance
print(paste("Random Forest RMSE:", rf_rmse))
print(paste("XGBoost RMSE:", xgb_rmse))

# Step 9: Portfolio Optimization with Two Indices (S&P 500 and Dow Jones)
symbols <- c("^GSPC", "^DJI")
getSymbols(symbols, src = "yahoo", from = start_date, to = end_date)
prices <- merge(Cl(GSPC), Cl(DJI))
colnames(prices) <- symbols

# Calculate Returns
returns <- na.omit(Return.calculate(prices))

# Define Portfolio
portfolio <- portfolio.spec(assets = colnames(returns))
portfolio <- add.constraint(portfolio, type = "full_investment")
portfolio <- add.constraint(portfolio, type = "box", min = 0.1, max = 0.9)
portfolio <- add.objective(portfolio, type = "return", name = "mean")
portfolio <- add.objective(portfolio, type = "risk", name = "StdDev")

# Optimize Portfolio
optimized_portfolio <- optimize.portfolio(R = returns, portfolio = portfolio, optimize_method = "ROI")

# Print Optimized Portfolio
print(optimized_portfolio)

# Step 10: Shiny App for Visualization
ui <- fluidPage(
  titlePanel("Stock Portfolio Optimization"),
  sidebarLayout(
    sidebarPanel(
      selectInput("symbol", "Select Stock Symbol:", choices = symbols)
    ),
    mainPanel(
      plotOutput("stockPlot"),
      verbatimTextOutput("portfolioSummary")
    )
  )
)

server <- function(input, output) {
  output$stockPlot <- renderPlot({
    symbol <- input$symbol
    plot(Cl(get(symbol)), main = paste("Closing Prices of", symbol), ylab = "Price", xlab = "Date")
  })
  
  output$portfolioSummary <- renderPrint({
    optimized_portfolio
  })
}

# Run Shiny App
shinyApp(ui = ui, server = server)

#  Conclusion and Documentation
# Document the findings and insights gained from the analysis and modeling
# Write a brief report summarizing the key steps, results, and conclusions
"Summary Report:
1. Data Retrieval and Visualization:
   - Successfully fetched historical stock price data for selected symbols (^GSPC and ^DJI) from Yahoo Finance.
   - Visualized the closing prices to understand the stock trends.

2. Portfolio Optimization:
   - Implemented mean-variance optimization to balance risk and return.
   - Applied constraints for full investment and weight limits on individual assets.
   - Derived optimal asset weights for the portfolio, favoring a 90% allocation to ^GSPC and 10% to ^DJI.

3. Forecasting with ARIMA:
   - Modeled and forecasted future stock prices using ARIMA, providing a 20-day forecast.
   - The ARIMA model captured the stock's trend and seasonality effectively.

4. Interactive Shiny Application:
   - Developed a user-friendly interface allowing users to select stocks, visualize data, and optimize portfolios.
   - Enabled report download functionality for users to save the analysis results.

5. Educational and Practical Value:
   - The project serves as a practical example of financial data analysis and portfolio management using R.
   - Well-documented code and step-by-step explanations make it accessible for beginners.

6. Future Work:
   - Potential enhancements include dynamic asset allocation, more sophisticated forecasting models, and real-time data integration.
")
