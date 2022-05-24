# Author: Terry Keyrouz - https://github.com/terrytyk77

# Import the necessary modules
import functions as my_func

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Constants
TEST_SIZE_SCALE = 0.1  # How much from 0 to 1, is attributed to the test set from all the data
USELESS_COLUMNS = ['id', 'wind_speed', 'atmo_opacity']
ATTRIBUTE_COLUMNS = ['sol', 'ls', 'month', 'terrestrial_day', 'terrestrial_month', 'terrestrial_year']
GOAL_COLUMNS = ['min_temp', 'max_temp', 'pressure']

MODELS = {
    'KNeighbors Regressor': KNeighborsRegressor(n_neighbors=2),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=500),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Neural Networks Regressor': MLPRegressor(random_state=1, max_iter=1000),
    'Ridge': Ridge()
}

# Console beauty
np.set_printoptions(threshold=10)

# Create data frame from the .csv file
Mars_DF = pd.read_csv('mars-weather-dataset.csv')

# Correlation Matrix Before Pre-processing
print(f'\n{my_func.Colors.YELLOW}Before Pre-processing Data Shape: {my_func.Colors.RESET}{Mars_DF.shape}')
print(f'{Mars_DF.info()}\n')
my_func.plot_correlation_matrix(plt, 'Before Pre-processing Correlation Matrix', Mars_DF)

# As we can see before pre-processing any data, we have 3 object field
# Which means they are not very relevant in the state they are for the model
# We will have to convert them to something more meaningful for better prediction
# As we can see the wind_speed has 0 non-null count which marks it as safe to remove

# Pre-processing the data:
# Drop id column, because it's just an irrelevant numerical value, that doesn't relate to any field
# Drop wind_speed column, as they are all NaN which means they are not contributing to anything
# Drop atmo_opacity column, because they are all 100% sunny, which renders this column irrelevant for our prediction
# Removing all these columns is removing unwanted complexity.
Mars_DF = Mars_DF.drop(columns=USELESS_COLUMNS)

# Remove incomplete data, the number of incomplete rows is minimal and will improve the robustness of our model
Mars_DF.dropna(inplace=True)

# Date was a string field, converting it to datetime will be usable to generate more relevant information
# Getting the day, month, and year as Integers will be more useful to predict the weather than a simple string
Date = Mars_DF['terrestrial_date']
Date = pd.to_datetime(Date)
Mars_DF['terrestrial_day'] = Date.dt.day
Mars_DF['terrestrial_month'] = Date.dt.month
Mars_DF['terrestrial_year'] = Date.dt.year

# Drop the column 'terrestrial_date' as it won't be used anymore
Mars_DF = Mars_DF.drop(columns=['terrestrial_date'])

# Convert the 'month' column into an int, by removing the word 'Month ' and converting it afterwards,
# this will render the field 'month' more useful for the prediction
Mars_DF['month'] = np.int64(Mars_DF['month'].str.replace('Month ', ''))

# Correlation Matrix After Pre-processing
print(f'{my_func.Colors.GREEN}After Pre-processing Data Shape: {my_func.Colors.RESET}{Mars_DF.shape}')
print(f'{Mars_DF.info()}\n')
my_func.plot_correlation_matrix(plt, 'After Pre-processing Correlation Matrix', Mars_DF)

# Create training sets
X = Mars_DF.drop(columns=GOAL_COLUMNS)
Y = Mars_DF.drop(columns=ATTRIBUTE_COLUMNS)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE_SCALE)

# Test all the models
for key, model in MODELS.items():
    model.fit(X_train, Y_train)
    Y_predictions = model.predict(X_test)

    print(f'{my_func.Colors.CYAN}Algorithm: {my_func.Colors.RESET}{key}')
    print(f'{my_func.Colors.YELLOW}Actual Data:\n{my_func.Colors.RESET}{Y_test}\n')
    print(f'{my_func.Colors.GREEN}Predicted Data:\n{my_func.Colors.RESET}{Y_predictions}\n')

    mean_squared_error_score = mean_squared_error(Y_test, Y_predictions)
    root_mean_squared_error_score = sqrt(mean_squared_error_score)
    mean_absolute_error_score = mean_absolute_error(Y_test, Y_predictions)
    r_squared_coefficient_score = r2_score(Y_test, Y_predictions)

    print(f'{my_func.Colors.GREEN}MSE: {my_func.Colors.RESET}{mean_squared_error_score:.3f}')
    print(f'{my_func.Colors.GREEN}RMSE: {my_func.Colors.RESET}{root_mean_squared_error_score:.3f}')
    print(f'{my_func.Colors.GREEN}MAE: {my_func.Colors.RESET}{mean_absolute_error_score:.3f}')
    print(f'{my_func.Colors.GREEN}RSC: {my_func.Colors.RESET}{r_squared_coefficient_score:.3f}\n')
