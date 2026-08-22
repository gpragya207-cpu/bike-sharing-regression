
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
# Load the hourly data and check the data
hour_df= pd.read_csv('hour.csv')
print(hour_df.shape)
print(hour_df.columns)
hour_df.head()
# %%
# check for missing values
print(hour_df.isnull().sum())
# %%
# check data types
print(hour_df.dtypes)

# exploring the dataset 
# distribution of rentals using cnt column
# %%
hour_df['cnt'].plot(kind= 'hist', bins= 40, title= 'Distribution of hourly rentals', color= 'blue')
plt.xlabel('Rentals per hour')
plt.ylabel('Frequency')
plt.savefig('plots/cnt_distribution.png')
plt.show() 

# average rentals per hour of the day (line graph since we are looking avg rentals over time)
hour_df.groupby('hr')['cnt'].mean().plot(kind= 'line', title= 'Average rentals by hour of the day', color= 'orange')
plt.xlabel('Hour')
plt.ylabel('Avg Rentals')
plt.savefig('plots/avg_rental_per_hour.png')
plt.show()
# rentals go up during the day and peak around 8 am and 5 pm, which is expected as these are the peak commuting hours.

# temperature vs rentals
hour_df.plot(kind= 'scatter', x= 'temp', y= 'cnt', alpha=0.1 , title= 'Temperature vs Rentals', color= 'green')
plt.xlabel('Temperature (normalized)')
plt.savefig('plots/temp_vs_rentals.png')
plt.show()
# scatter plot shows a positiive correlation between temperature and rentals, which is expected as people are more likely to rent bikes in warmer weather.

# rentals by weather situation
hour_df.groupby('weathersit')['cnt'].mean().plot(kind='bar', title='Avg rentals by weather condition')
plt.xlabel('Weather situation (1=clear, 4=heavy rain/snow)')
plt.ylabel('Avg rentals')
plt.savefig('plots/weather_vs_rentals.png')
plt.show()
# similar to temperature, rentals are higher in clear weather and lower in bad weather conditions.

# humidity vs rentals (both continuous, so scatter plot)
hour_df.plot(kind='scatter', x='hum', y='cnt', alpha=0.1, title='Humidity vs Rentals')
plt.xlabel('Humidity (normalized)')
plt.ylabel('Rentals')
plt.savefig('plots/hum_vs_rentals.png')
plt.show()
# not much correlation between humidity and rentals, but there is a slight negative correlation, which is expected as people are less likely to rent bikes in high humidity.

# rentals by season (season is categorical so bar chart)
hour_df.groupby('season')['cnt'].mean().plot(kind='bar', title='Avg rentals by season')
plt.xlabel('Season (1=winter, 2=spring, 3=summer, 4=fall)')
plt.ylabel('Avg rentals')
plt.savefig('plots/rental_by_season.png')
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
plt.savefig('plots/temp_regression_line.png')
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
plt.savefig('plots/temp_residual_plot.png')
plt.show()
# residual fan out at higher temperatures and rather clustered at low temperatures (heteroscedasticity)
# %%

# multiple regression with temperature, humidity, and weather situation as predictors
# %%
# defining a function to split the hr column into categories for encoding later
# hour bucketed into time-of-day categories instead of used as a raw number (see README for why this mattered (fixed a sign-flip issue with weathersit))

def hour_cat(hr):
    if hr in [0, 1, 2, 3, 4, 5]:
        return 'Night'
    elif hr in [6, 7, 8, 9]:
        return 'Morning Rush' # as we noted in the earlier analysis
    elif hr in [10, 11, 12, 13, 14, 15]:
        return 'Midday'
    elif hr in [16, 17, 18, 19]:
        return 'Evening Rush' # as we noted in the earlier analysis
    else:
        return 'Evening'

hour_df['hr_cat'] = hour_df['hr'].apply(hour_cat)
model_df = hour_df[['temp', 'hum', 'weathersit', 'hr_cat', 'season', 'cnt']]
model_encoded = pd.get_dummies(model_df, columns=['weathersit', 'season', 'hr_cat'], drop_first=True) # drop first= True to avoid dummy variable trap
print(model_encoded.head())
# %%
from sklearn.linear_model import LinearRegression
#define the features and target variable
X= model_encoded.drop(columns= 'cnt')
Y = model_encoded.get('cnt')

# %%
# create and fit the linear regression model
model = LinearRegression()
model.fit(X, Y)

# %%
# look at the coefficients with respective feature names
coeff=pd.Series(model.coef_, index= X.columns)
print(coeff.sort_values(ascending=True))
print(f'Intercept: {model.intercept_}')
# the coefficients indicate the change in rentals for a unit change in the predictor variable, holding all other variables constant.
# temp has the highest positive coeff and hum has the highest negative coeff

# previous code showed which variables matter and in what direction; now checking how well 
# the model predicts on unseen data (train/test split)

from sklearn.model_selection import train_test_split
# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# create a new linear regression model and fit it to the training data (to keep the test data unseen)
# 80% of the data is used for training and 20% for testing 

eval_model = LinearRegression()
eval_model.fit(X_train, Y_train) # fit the model to the training data

# %%
# small check to see if the model is working
print(eval_model.coef_)
# all 12 columns from coeff present) 

# %%
# %%
from sklearn.metrics import r2_score

Y_pred = eval_model.predict(X_test)
r2 = r2_score(Y_test, Y_pred)
print(f"R²: {r2}")
# the multiple regression model explains approximately 54% of the variance in hourly bike rentals (R² = 0.54)
# rest of the variation remains unexplained likely due to other factors not included in the model

# testiing the model on a single example to see how it performs
# %%
print(X.columns.tolist())
# %%
# creating a dictionary with key value pairs in the same order as the columns
conditions= {
    'temp': 0.15, #since its normalised
    'hum': 0.85,
    'weathersit_2': 0,
    'weathersit_3': 1,
    'weathersit_4': 0,
    'season_2': 0,
    'season_3': 0,
    'season_4': 0, #season 1 (winter) is the base case
    'hr_cat_Evening Rush': 1,
    'hr_cat_Midday': 0,
    'hr_cat_Morning Rush': 0,
    'hr_cat_Night': 0 # evening is the base case
}
cond_df= pd.DataFrame([conditions], index=[0])
print(cond_df)

# %%
# quick check to see if the column order is correct
scenario = cond_df[X.columns.tolist()]
print(len(scenario.columns) ==len(X.columns)) # evaluated to True
# %%

# predict the rentals for the given scenario
predicted_rentals_model = eval_model.predict(scenario)
print(f"Predicted rentals for the given scenario: {predicted_rentals_model[0]}")
# %%
check = hour_df[(hour_df['season'] == 1) & (hour_df['weathersit'] == 3) & (hour_df['hr'].between(16,19))]
print(check.shape[0])
print(check['cnt'].mean(), len(check)) # came out to be around 118
# this gap likely reflects the model's linear, additive structure 
# it assumes each factor's effect (temperature, humidity, weather, time-of-day) contributes independently, 
# but in reality, rush-hour demand may be less weather-sensitive

