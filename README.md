# Regression-Bike-Sharing-Demand-Prediction

This code snippet represents a machine learning project for bike sharing demand prediction. Let's go through each section and understand what it does:

A) Import Libraries:
This section imports the necessary libraries for data manipulation, visualization, and model training. The imported libraries include:
- `pandas`: Used for data manipulation and aggregation.
- `numpy`: Used for computationally efficient operations.
- `matplotlib.pyplot` and `seaborn`: Used for data visualization.
- `datetime`: Used for working with date and time.
- Various modules from the `sklearn` library, including preprocessing, linear regression, decision tree regressor, random forest regressor, gradient boosting regressor, model selection, and metrics.

B) Dataset Loading:
This section mounts the Google Drive and reads the dataset file from the specified path using `pd.read_csv()`. The dataset file is assumed to be stored in the Google Drive.

C) Dataset Information:
This section provides information about the dataset. It displays the data types and non-null counts for each column in the dataset.

D) There are no duplicate and null values:
This statement indicates that there are no duplicate or null values in the dataset.

E) Data Wrangling Code:
This section performs data preprocessing and feature engineering tasks. The code includes the following steps:
- Converting the 'Date' column to a datetime object.
- Extracting date and time features such as year, month, day, hour, and day of the week from the 'datetime' column.
- Dropping the 'datetime' column as it is no longer needed.
- Converting categorical columns ('Seasons', 'Holiday', 'Functioning Day') to the categorical data type.
- Printing the unique values for each categorical variable.

Overall, this code snippet imports the necessary libraries, loads the dataset, provides information about the dataset, and performs data preprocessing and feature engineering steps to prepare the dataset for machine learning modeling.
The code snippet provided focuses on data wrangling, which involves transforming and preparing the dataset for analysis. Let's go through each step:

1. Convert 'datetime' column to datetime object:
   - The code uses `pd.to_datetime()` function to convert the 'Date' column in the 'data' DataFrame to a datetime object.
   - The result is stored in a new column named 'datetime' within the DataFrame.

2. Extract date and time features:
   - The code extracts various date and time features from the 'datetime' column using the `dt` accessor provided by pandas.
   - The extracted features include 'year', 'month', 'day', 'hour', and 'day_of_week', which represent the corresponding parts of the datetime.
   - Each feature is stored as a new column in the 'data' DataFrame.

3. Drop unnecessary columns:
   - The code drops the 'datetime' column since the extracted date and time features have already been stored in separate columns.
   - This step removes the redundant 'datetime' column from the DataFrame using the `drop()` function with the 'datetime' column name and specifying `axis=1` to indicate column-wise operation.
   - The `inplace=True` argument is used to modify the 'data' DataFrame directly.

4. Convert 'season' column to categorical data type:
   - The code converts the 'Seasons' column in the 'data' DataFrame to the categorical data type using `astype('category')`.
   - This conversion allows treating the 'Seasons' column as a categorical variable, which can be useful for certain types of analysis or modeling.

5. Convert 'holiday', 'workingday', and 'weather' columns to categorical data type:
   - Similar to the previous step, this code converts the 'Holiday' and 'Functioning Day' columns in the 'data' DataFrame to the categorical data type.
   - This conversion is performed using `astype('category')` for each column.

6. Check unique values for categorical variables:
   - The code prints the unique values of each categorical variable ('Seasons', 'Holiday', 'Functioning Day') in the 'data' DataFrame.
   - This step helps to verify the distinct categories present in each categorical column.

Overall, this data wrangling code converts the 'Date' column to a datetime object, extracts relevant date and time features, drops unnecessary columns, converts specific columns to categorical data type, and provides insights into the unique categories of categorical variables in the dataset. These steps help prepare the dataset for further analysis or modeling tasks.

1. Chart 1: Distribution of Target Variable
- This chart was chosen to visualize the distribution of the target variable, which is the 'Rented Bike Count'.
- The histogram with a kernel density estimation (KDE) curve provides an overview of the distribution and helps identify any skewness or patterns in the data.
- Insights: The chart shows the distribution of bike rentals, indicating the frequency of different rental counts. It can reveal if the data is skewed, bimodal, or normally distributed.

2. Chart 2: Count of Bike Rentals per Season
- This bar chart was chosen to compare the count of bike rentals across different seasons.
- Each bar represents a season, and the height of the bar represents the count of bike rentals.
- Insights: The chart allows us to compare the rental counts among seasons. It helps identify which season has the highest or lowest bike rental demand.

3. Chart 3: Count of Bike Rentals per Month
- This line plot was chosen to show the count of bike rentals over different months.
- The x-axis represents the months, and the y-axis represents the count of bike rentals.
- Insights: The chart helps observe the trend of bike rentals over months, providing insights into any seasonal patterns or changes in demand.

4. Chart 4: Subplots for Distribution and Correlation Analysis
- This subplot arrangement consists of four individual plots for different purposes.
- The first subplot displays the distribution of the target variable ('Rented Bike Count') using a histogram.
- The other three subplots show the scatter plots between the target variable and different features: temperature, humidity, and windspeed.
- Insights: The subplot arrangement allows for simultaneous analysis of the target variable's distribution and its correlations with important factors such as temperature, humidity, and windspeed. It helps identify any relationships or patterns between these variables.

5. Chart 5: Count of Bike Rentals per Hour
- This bar chart was chosen to visualize the count of bike rentals based on different hours of the day.
- Each bar represents an hour, and the height of the bar represents the count of bike rentals.
- Insights: The chart shows the hourly variations in bike rental demand, providing insights into peak hours or periods of high and low demand throughout the day.

6. Chart 6: Count of Bike Rentals on Holidays or Not
- This bar chart was chosen to compare the count of bike rentals on holidays versus non-holidays, with further differentiation by seasons using different colors.
- Each bar represents a category (holiday or non-holiday), and the height of the bar represents the count of bike rentals.
- Insights: The chart allows for a comparison between bike rental counts on holidays and non-holidays. Additionally, it shows how the rental counts vary across different seasons for each category. This can help identify the impact of holidays and seasons on bike rental demand.

These charts were selected based on the specific analysis goals, such as understanding the distribution of the target variable, exploring patterns across different seasons or months, examining correlations with other variables, and identifying trends or patterns within specific time periods or categories.
7) Chart 7: Count of Bike Rentals by Seasons
- This bar chart was chosen to compare the count of bike rentals across different seasons, with further differentiation by hour of the day using different colors.
- Each bar represents a season, and the height of the bar represents the count of bike rentals. The bars are further divided into segments based on the hour of the day.
- Insights: The chart provides a comprehensive view of bike rental counts by seasons and their distribution across different hours. It helps identify any variations in rental counts between seasons and within different hours of the day.

8) Chart 8: Count of Bike Rentals by Temperature
- This line plot was chosen to show the relationship between bike rental counts and temperature.
- The x-axis represents temperature values, and the y-axis represents the count of bike rentals.
- Insights: The chart helps observe the trend in bike rentals based on temperature. It can indicate if there is any correlation between temperature and rental counts, such as higher rentals during certain temperature ranges.

9) Chart 9: Count of Bike Rentals by Humidity
- This line plot was chosen to show the relationship between bike rental counts and humidity levels.
- The x-axis represents humidity values, and the y-axis represents the count of bike rentals.
- Insights: The chart helps identify any correlation between humidity and bike rental counts. It provides insights into how humidity levels may affect rental demand.

10) Chart 10: Count of Bike Rentals by Wind Speed
- This line plot was chosen to show the relationship between bike rental counts and wind speed.
- The x-axis represents wind speed values, and the y-axis represents the count of bike rentals.
- Insights: The chart helps observe the trend in bike rentals based on wind speed. It allows for the analysis of any correlation between wind speed and rental counts, indicating if windy conditions impact rental demand.

11) Chart 11: Bike Rentals by Temperature and Humidity
- This scatter plot was chosen to visualize the relationship between bike rentals, temperature, and humidity. The data points are differentiated by seasons using different colors.
- The x-axis represents temperature values, the y-axis represents the count of bike rentals, and the hue represents different seasons.
- Insights: The chart helps analyze the combined effect of temperature and humidity on bike rentals. It can reveal any patterns or relationships between these two factors and rental counts, further differentiated by seasons.

12) Chart 12: Bike Rentals by Temperature and Wind Speed
- This scatter plot was chosen to visualize the relationship between bike rentals, temperature, and wind speed. The data points are differentiated by seasons using different colors.
- The x-axis represents temperature values, the y-axis represents wind speed, and the hue represents different seasons.
- Insights: The chart allows for the examination of the interplay between temperature, wind speed, and bike rentals. It helps identify any trends or patterns in rental counts concerning these two factors, with further differentiation by seasons.

13) Chart 13: Count of Bike Rentals per Month
- This bar chart was chosen to compare the count of bike rentals across different months.
- Each bar represents a month, and the height of the bar represents the count of bike rentals.
- Insights: The chart provides an overview of bike rental counts per month. It helps identify any seasonal variations or patterns in rental demand throughout the year.

14) Correlation Heatmap
- This heatmap was created to visualize the correlation between different variables in the dataset.
- Each cell represents the correlation coefficient between two variables, and the color represents the strength and direction of the correlation.
- Insights: The heatmap allows for the identification of relationships and dependencies between variables. It helps understand which variables are strongly correlated or inversely related to each other, providing insights into potential factors influencing bike rental counts.

15) Pair Plot

1) Hypothetical Statement - 1:
The purpose of this hypothesis test is to determine if there is a significant difference in the mean number of bike rentals between functional and non-functional days. The test is performed using the t-test for independent samples.

- The null hypothesis (H0) states that there is no significant difference in the mean number of bike rentals between functional and non-functional days.
- The alternative hypothesis (H1) states that there is a significant difference in the mean number of bike rentals between functional and non-functional days.

The test is conducted by dividing the data into two groups based on the 'Functioning Day' variable (assuming it is binary with values 0 and 1). The t-test is then performed, and the resulting p-value is compared to a significance level (usually 0.05).

In this case, the p-value is found to be greater than 0.05, indicating that there is not enough evidence to reject the null hypothesis. Therefore, the conclusion is that there is no significant difference in the mean number of bike rentals between functional and non-functional days.

2) Hypothetical Statement - 2:
This hypothesis test aims to determine if there is a significant difference in the mean number of bike rentals between holidays and non-holidays. The test is performed using the t-test for independent samples.

- The null hypothesis (H0) states that there is no significant difference in the mean number of bike rentals between holidays and non-holidays.
- The alternative hypothesis (H1) states that there is a significant difference in the mean number of bike rentals between holidays and non-holidays.

The data is divided into two groups based on the 'Holiday' variable, where one group represents bike rentals on holidays and the other group represents rentals on non-holidays. The t-test is then conducted, and the resulting p-value is compared to the significance level.

In this case, the p-value is found to be greater than 0.05, indicating that there is not enough evidence to reject the null hypothesis. Therefore, the conclusion is that there is no significant difference in the mean number of bike rentals between holidays and non-holidays.

3) Hypothetical Statement - 3:
This hypothesis test aims to determine if there is a significant difference in the mean number of bike rentals between daytime and nighttime. The test is performed using the t-test for independent samples.

- The null hypothesis (H0) states that there is no significant difference in the mean number of bike rentals between daytime and nighttime.
- The alternative hypothesis (H1) states that there is a significant difference in the mean number of bike rentals between daytime and nighttime.

The data is divided into two groups based on the time of day, with one group representing daytime rentals (hours between 6 and 17) and the other group representing nighttime rentals (hours outside of this range). The t-test is then conducted, and the resulting p-value is compared to the significance level.

In this case, the p-value is found to be greater than 0.05, indicating that there is not enough evidence to reject the null hypothesis. Therefore, the conclusion is that there is no significant difference in the mean number of bike rentals between daytime and nighttime.
H) Feature Engineering & Data Pre-processing:

1) Handling Outliers & Outlier Treatments:
This code snippet demonstrates how to handle outliers in the 'Rented Bike Count' variable. The box plot is used to visualize the distribution of the variable. In this case, there are a few outliers present, but they are not removed at this stage as they could potentially contain important information.

2) Encode Categorical Variables:
In this code snippet, categorical variables are encoded using one-hot encoding. The `pd.get_dummies()` function is applied to the 'Seasons', 'month', 'day_of_week', 'Hour', 'Holiday', and 'Functioning Day' columns of the dataset. This process converts categorical variables into numerical binary columns, making them suitable for machine learning algorithms that require numerical inputs.

3) Manipulate Features to Minimize Feature Correlation and Create New Features:
This code snippet involves manipulating features to minimize correlation between them and create new features. The 'day' and 'Date' columns are dropped from the 'train_df' DataFrame as they are considered unwanted variables for the analysis.

4) Transform Your Data:
Here, a logarithmic transformation is applied to the 'Rented Bike Count' variable using `np.log1p()`. This transformation is commonly used to normalize the distribution of skewed variables and can help improve the performance of certain machine learning models.

5) Count the Number of Instances in Each Class:
This snippet calculates the number of instances (observations) in each class of the 'Rented Bike Count' variable. It uses the `value_counts()` function to count the occurrences of each unique value and then calculates the proportion of each class by dividing the counts by the total length of the dataset.

6) Handling Imbalanced Dataset (If Needed):
If the dataset is imbalanced, this code snippet demonstrates how to upsample the minority class using the `resample()` function from scikit-learn's `utils` module. It separates the majority class and minority class, performs upsampling on the minority class by randomly duplicating samples with replacement, and then combines the upsampled minority class with the majority class. This helps balance the class distribution and address potential bias in the model.

I) Data Scaling:
This code snippet showcases data scaling using the StandardScaler from scikit-learn's preprocessing module. The continuous variables 'Temperature(°C)', 'Wind speed (m/s)', and 'Humidity(%)' are selected and scaled using `StandardScaler`. Scaling transforms the variables to have zero mean and unit variance, ensuring that they are on the same scale and preventing features with larger values from dominating the model.

J) Data Splitting:
This code snippet demonstrates how to split the data into training and testing sets using the train_test_split function from scikit-learn's model_selection module. The feature matrix 'X' is assigned all columns except for 'Rented Bike Count', which is assigned to the target variable 'y'. The data is split into training and testing sets with a test size of 0.2 (20% of the data) and a random state of 42 for reproducibility. The resulting splits are stored in 'X_train', 'X_test', 'y_train', and 'y_test'.
In the provided code, three different machine learning models are implemented and evaluated for a regression task. Here's an explanation of each part:

K) ML Model Implementation:

a) ML Model - 1: RandomForestRegressor

- Random Forest Regressor is a machine learning model based on the random forest algorithm for regression tasks.
- In the code, the dataset is split into training and testing sets using the `train_test_split` function from `sklearn.model_selection`. The test set size is set to 30% of the data.
- A Random Forest Regressor object is created with 100 estimators (decision trees) using the `RandomForestRegressor` class from `sklearn.ensemble`.
- The model is trained on the training data using the `fit` method.
- Predictions are made on the testing data using the `predict` method.
- Mean squared error (MSE) and R-squared are calculated to evaluate the model's performance using the `mean_squared_error` and `r2_score` functions from `sklearn.metrics`.
- The MSE and R-squared values are printed to assess the model's performance.

b) ML Model - 2: XGBRegressor

- XGBRegressor is a machine learning model based on the gradient boosting algorithm using the XGBoost library.
- Similar to the Random Forest Regressor, the dataset is split into training and testing sets.
- An XGBRegressor object is created with 100 estimators using the `XGBRegressor` class from `xgboost`.
- The model is trained on the training data.
- Predictions are made on the testing data.
- MSE and R-squared are calculated to evaluate the model's performance.
- The evaluation metrics are plotted using a bar chart.

c) ML Model - 3: GradientBoostingRegressor

- GradientBoostingRegressor is another machine learning model based on the gradient boosting algorithm from the `sklearn.ensemble` module.
- The dataset is split into training and testing sets.
- A GradientBoostingRegressor object is created with 100 estimators and a learning rate of 0.1.
- The model is trained on the training data.
- Predictions are made on the testing data.
- MSE, mean absolute error (MAE), and R-squared are calculated to evaluate the model's performance.
- The evaluation metrics are printed and plotted using a bar chart.

For all three models, the evaluation metrics provide an indication of how well the models perform. Lower values of MSE and MAE indicate better predictive accuracy, while a higher R-squared value (closer to 1) indicates a better fit of the model to the data. The bar charts visualize the evaluation metric scores for easy comparison between the models.

