
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the hourly data and check the data
hour_df= pd.read_csv('hour.csv')
print(hour_df.shape)
print(hour_df.columns)
hour_df.head()

# check for missing values
print(hour_df.isnull().sum())

# check data types
print(hour_df.dtypes)


