
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

# exploring the dataset 
# distribution of rentals using cnt column
# %%
hour_df['cnt'].plot(kind= 'hist', bins= 40, title= 'Distribution of hourly rentals', color= 'blue')
plt.xlabel('Rentals per hour')
plt.ylabel('Frequency')
plt.show() 

# average rentals per hour of the day (line graph since we are looking avg rentals over time)
hour_df.groupby('hr')['cnt'].mean().plot(kind= 'line', title= 'Average rentals by hour of the day', color= 'orange')
plt.xlabel('Hour')
plt.ylabel('Avg Rentals')
plt.show()
# rentals go up during the day and peak around 8 am and 5 pm, which is expected as these are the peak commuting hours.

# temperature vs rentals
hour_df.plot(kind= 'scatter', x= 'temp', y= 'cnt', alpha=0.1 , title= 'Temperature vs Rentals', color= 'green')
plt.xlabel('Temperature (normalized)')
plt.show()
# scatter plot shows a positiive correlation between temperature and rentals, which is expected as people are more likely to rent bikes in warmer weather.

# rentals by weather situation
hour_df.groupby('weathersit')['cnt'].mean().plot(kind='bar', title='Avg rentals by weather condition')
plt.xlabel('Weather situation (1=clear, 4=heavy rain/snow)')
plt.ylabel('Avg rentals')
plt.show()
# similar to temperature, rentals are higher in clear weather and lower in bad weather conditions.

# humidity vs rentals (both continuous, so scatter plot)
hour_df.plot(kind='scatter', x='hum', y='cnt', alpha=0.1, title='Humidity vs Rentals')
plt.xlabel('Humidity (normalized)')
plt.ylabel('Rentals')
plt.show()
# not much correlation between humidity and rentals, but there is a slight negative correlation, which is expected as people are less likely to rent bikes in high humidity.

# rentals by season (season is categorical so bar chart)
hour_df.groupby('season')['cnt'].mean().plot(kind='bar', title='Avg rentals by season')
plt.xlabel('Season (1=winter, 2=spring, 3=summer, 4=fall)')
plt.ylabel('Avg rentals')
plt.show()
# rentals are highest in summer and lowest in winter, which is expected as people are more likely to rent bikes in warmer weather.

# season and temperature might be correlated due to the underlying seasonal warmth pattern 

# %%
# initial simple regression on temperature and rentals

def standard_units(col):
    return (col - col.mean()) / np.std(col)
def calculate_r(df, x, y):
    x_su = standard_units(df.get(x))
    y_su = standard_units(df.get(y))
    return (x_su * y_su).mean()
def slope(df, x, y):
    r = calculate_r(df, x, y)
    return r * np.std(df.get(y)) / np.std(df.get(x))

def intercept(df, x, y):
    return df.get(y).mean() - slope(df, x, y) * df.get(x).mean()
# %%
temp_r = calculate_r(hour_df, 'temp', 'cnt')
temp_slope = slope(hour_df, 'temp', 'cnt')
temp_intercept = intercept(hour_df, 'temp', 'cnt')

# regression line in original units
def predicted(df, x, y):
    m = slope(df, x, y)
    b = intercept(df, x, y)
    return m * df.get(x) + b

# %%
print(f'correlation: {temp_r}', f'slope: {temp_slope}', f'intercept: {temp_intercept}')
# regression intercepts cannot be interpreted at extremes (eg. here rentals at 0 temperature) since we do not have data at those extremes.
# %%
hour_df.get(['temp', 'cnt']).plot(kind='scatter', x='temp', y='cnt', alpha=0.1, title='Temperature vs Rentals with Regression Line')
plt.xlabel('Temperature (normalized)')
plt.ylabel('Rentals')
xs= np.linspace(hour_df.get('temp').min(), hour_df.get('temp').max(), 100)
plt.plot(xs, temp_slope* xs + temp_intercept, color='red', label= 'Regression Line')
plt.legend()
plt.show()
# %%
# plotting the residuals to check for patterns with RMSE
predict_temp = predicted(hour_df, 'temp', 'cnt')
residuals = hour_df.get('cnt') - predict_temp
plt.scatter(hour_df.get('temp'), residuals, alpha=0.1, label='Residuals')
plt.axhline(y=0, color='red', linestyle='--', label='y=0')
plt.title('Residuals of Temperature vs Rentals Regression')
plt.xlabel('Temperature (normalized)')
plt.ylabel('Residuals')
plt.legend()
plt.show()
# residual fan out at higher temperatures and rather clustered at low temperatures (heteroscedasticity)
# %%
