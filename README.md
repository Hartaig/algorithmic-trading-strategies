# trading-strategy-next-day-close-prediction
Predicting based on prior technical indicators and fundamentals if a stock will close positive or negative 


Longitudinal Data Aggregation notebook contains the necessary classes and methods to aggregate stock data daily. 
--Everyday fundemental and technical indicators of stock are pulled and combined into a longitudinal data set
--The goal is then to predict the probability at stock will close positive the next day given all data prior to that day. 

Model of choice will be a mixed effect logistic regression with random intercepts on stock and sector.

Feature engineering will consists of principle components to reduce colinearity 
