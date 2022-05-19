# Import the necessary modules
import functions as my_func

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Constants
TEST_SIZE_SCALE = 0.25  # How much from 0 to 1, is attributed to the test set from all the data
USELESS_COLUMNS = ["id", "wind_speed", "atmo_opacity"]
ATTRIBUTE_COLUMNS = ["sol", "ls", "month", "terrestrial_day", "terrestrial_month", "terrestrial_year"]
GOAL_COLUMNS = ["min_temp", "max_temp", "pressure"]

MODELS = {
    "KNeighbors Regressor ": KNeighborsRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    'Ridge': Ridge(),
    "Neural Networks Regressor": MLPRegressor(random_state=1, max_iter=5000)
}

# Create data frame from the .csv file
Mars_DF = pd.read_csv('mars-weather-dataset.csv')

# Correlation Matrix Before Pre-processing
print(Mars_DF.info())
print('\033[93m', 'Default Data Shape:', Mars_DF.shape, '\033[0m')
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

# Remove incomplete data, the number of incomplete rows are minimal and will improve the robustness of our model
Mars_DF.dropna(inplace=True)

# Date was a string field, converting it to datetime will be usable to generate more relevant information
# Getting the day, month, and year as Integers will be more useful to predict the weather than a simple string
Date = Mars_DF['terrestrial_date']
Date = pd.to_datetime(Date)
Mars_DF["terrestrial_day"] = Date.dt.day
Mars_DF["terrestrial_month"] = Date.dt.month
Mars_DF["terrestrial_year"] = Date.dt.year

# Drop the column 'terrestrial_date' as it won't be used anymore
Mars_DF = Mars_DF.drop(columns=['terrestrial_date'])

# Convert the 'month' column into an int, by removing the word 'Month ' and converting,
# this will render the field 'month' more useful for the prediction
Mars_DF["month"] = np.int64(Mars_DF["month"].str.replace('Month ', ''))

# Correlation Matrix After Pre-processing
print(Mars_DF.info())
print('\033[92m', 'After Pre-processing Data Shape:', Mars_DF.shape, '\033[0m', "\n")
my_func.plot_correlation_matrix(plt, 'After Pre-processing Correlation Matrix', Mars_DF)

# Create training sets
X = Mars_DF.drop(columns=GOAL_COLUMNS)
Y = Mars_DF.drop(columns=ATTRIBUTE_COLUMNS)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE_SCALE)

# Test all the models
for key, model in MODELS.items():
    model.fit(X_train, Y_train)

#     # Get the set that gave the maximum result
#     x_train, x_test, y_train, y_test = TRAIN_TEST[index_max]
#
#     # Compute predicted probabilities
#     y_pred_prob = model.predict_proba(x_test)[:, 1]
#
#     # Generate ROC curve values
#     fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
#
#     # Plot ROC curve
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.plot(fpr, tpr)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(key + ' ROC Curve')
#     plt.show()
