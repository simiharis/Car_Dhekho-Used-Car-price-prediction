# Car_Dhekho-Used-Car-price-prediction
A machine learning project to analyse  and process the car data and build a machine learning model to predict the price
Objective

The Used Car Price Prediction Project aims to develop a predictive model that can estimate the resale value of used cars based on a variety of features such as brand, model, mileage, age, engine size, and other car attributes. This project leverages historical data to identify key factors affecting car prices and applies machine learning techniques to create an accurate and robust prediction model.

Steps

Data Collection and Preprocessing

Data Collection: The project begins with gathering data from reliable sources, such as car sales websites, dealership records, or online marketplaces. The data typically includes attributes like:

Car Features: Brand, model, body type, fuel type, engine capacity, transmission type.
Usage Metrics: Mileage (kilometers driven), year of manufacture.
Other Factors: Insurance validity, location (city or region), owner history.

Data Cleaning and Preprocessing:

Handling Missing Values: Missing data for specific fields (e.g., insurance validity) is imputed or dropped based on the significance of the attribute.
Feature Engineering: Additional useful features, like the car’s age (calculated from the year of manufacture) or price per kilometer driven, may be created.
Encoding Categorical Variables: Categorical variables such as brand, model, fuel type, and transmission are encoded (e.g., using one-hot encoding) to be compatible with machine learning models.
EDA is conducted to gain insights into the data, identify trends, and understand the factors that significantly affect car prices. Examples of analyses performed:

Price Distribution by Brand and Model: 

Identify which brands and models retain higher resale values.

Impact of Car Age and Mileage on Price: 

Visualize how depreciation affects car prices as they age or accumulate mileage.

Fuel Type and Transmission Analysis: Understand consumer preferences and their impact on car prices.
Geographic Analysis: Determine if certain locations show higher demand or pricing trends for specific car types.
Model Selection and Training
Several machine learning algorithms are tested to find the best-performing model for this prediction task:

Linear Regression: A simple baseline model to capture the linear relationship between features and price.
Random Forest Regressor: A tree-based ensemble model that can capture non-linear relationships and interactions between features.
Gradient Boosting Machines (GBM): Models like XGBoost or LightGBM that are well-suited for regression tasks with structured data, providing higher accuracy and flexibility in handling outliers.
Decision trees
The models are trained and evaluated using standard regression metrics, such as:

Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual prices.
Root Mean Squared Error (RMSE): Penalizes larger errors, useful for identifying models with better overall accuracy.
Model Evaluation and Selection
The models are evaluated on a validation dataset, and the model with the best performance (lowest error and highest generalizability) is selected for deployment. Hyperparameter tuning is performed using techniques like Grid Search or Random Search to optimize model parameters for the best predictive performance.

Deployment and Application
Once trained, the final model is saved  using pickle  and integrated into a web application  for user interaction using streamlit. The deployment phase involves setting up a user-friendly interface where users can input car details to receive price predictions. 
