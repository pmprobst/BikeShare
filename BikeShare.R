# BikeShare Penalized Regression Analysis
# Load required libraries
install.packages("tidyverse")
install.packages("tidymodels")
install.packages("rpart")
install.packages("vroom")
install.packages("bonsai")
install.packages("lightgbm")
install.packages("dbarts")
install.packages("parsnip")
library(tidyverse)
library(tidymodels)
library(dplyr)
library(rpart)
library(vroom)
library(ranger)
library(bonsai)
library(lightgbm)
library(dbarts)
library(parsnip)

train_data <- vroom("data/train.csv") %>%
  select(-casual, -registered)
test_data <- vroom("data/test.csv")

# ===============================================================
# DATA CLEANING
# ===============================================================

my_recipe <- recipe(count ~ . ,data = train_data) %>%
  step_log(count, offset = 1, skip = TRUE) %>%
  # Recode weather "4" to "3" (combine rare weather conditions)
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  # Extract hour from datetime
  step_time(datetime, features = "hour") %>%
  
  # Day of week
  step_mutate(day_of_week = wday(datetime, label = TRUE)) %>%
  # Month of year
  step_mutate(month = month(datetime)) %>%
  # Weekend vs weekday
  step_mutate(is_weekend = ifelse(wday(datetime) %in% c(1, 7), 1, 0)) %>%
  # Rush hour indicators
  step_mutate(is_morning_rush = ifelse(hour(datetime) %in% 7:9, 1, 0)) %>%
  step_mutate(is_evening_rush = ifelse(hour(datetime) %in% 17:19, 1, 0)) %>%
  # Time of day categories
  step_mutate(time_of_day = case_when(
    hour(datetime) %in% 6:11 ~ "morning",
    hour(datetime) %in% 12:17 ~ "afternoon", 
    hour(datetime) %in% 18:22 ~ "evening",
    TRUE ~ "night"
  )) %>%
  
  # Temperature-humidity interaction
  step_mutate(temp_humidity = temp * humidity) %>%
  # Wind speed categories
  step_mutate(wind_category = case_when(
    windspeed < 10 ~ "calm",
    windspeed < 20 ~ "moderate",
    TRUE ~ "windy"
  )) %>%
  
  # Temperature bins
  step_discretize(temp, num_breaks = 4) %>%
  # Humidity bins
  step_discretize(humidity, num_breaks = 3) %>%
  # Quadratic terms for continuous variables 
  # (using atemp and windspeed to avoid conflicts)
  step_poly(atemp, degree = 2) %>%
  step_poly(windspeed, degree = 2) %>%
  
  
  # Distance from holidays
  step_mutate(days_from_holiday = abs(as.numeric(as.Date(datetime) - 
                                                   as.Date("2011-01-01")))) %>%
  # Special day indicators
  step_mutate(is_special_day = ifelse(holiday == 1 | workingday == 0, 1, 0)) %>%
  
  # Make weather a factor after recoding
  step_mutate(weather = factor(weather)) %>%
  # Make season a factor
  step_mutate(season = factor(season)) %>%
  # Make time_of_day a factor
  step_mutate(time_of_day = factor(time_of_day)) %>%
  # Make wind_category a factor
  step_mutate(wind_category = factor(wind_category)) %>%
  
  # Weather-temperature interaction (using numeric weather and temp)
  step_mutate(weather_temp = as.numeric(weather) * temp) %>%
  # Season-hour interaction (using numeric season)
  step_mutate(season_hour = as.numeric(season) * hour(datetime)) %>%
  # Season-weather interaction (using numeric values)
  step_mutate(season_weather = as.numeric(season) * as.numeric(weather)) %>%
  
  # Create dummy variables for all nominal predictors 
  #(encodes all categorical variables)
  step_dummy(all_nominal_predictors()) %>%
  # Remove highly correlated features to reduce multicollinearity
  step_corr(all_numeric_predictors(), threshold = 0.95) %>%
  # Remove near-zero variance predictors
  step_nzv(all_predictors()) %>%
  # Normalize all numeric predictors to put them on the same scale
  step_normalize(all_numeric_predictors()) %>%
  # Remove datetime column as it's not needed for modeling
  step_rm(datetime)

# ===============================================================
# GROW REGRESSION TREE
# ===============================================================
bart_model <- bart(trees = tune()) %>%
  set_engine("dbarts") %>%
  set_mode("regression")

# ===============================================================
# PENALIZED REGRESSION
# ===============================================================

# Establish a workflow
wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model)

# Create the parameter set automatically from the workflow and build a grid
param_set <- tune::extract_parameter_set_dials(wf)

# set grid
grid <- grid_regular(trees() ,levels = 2)

# Split data for Cross Validation
folds <- vfold_cv(train_data ,v = 4 ,repeats = 1)

#metrics_spec <- yardstick::metric_set(rmse)

# grow the forest
tuned <- tune_grid(
  wf,
  resamples = folds,
  grid = grid,
  metrics = metrics_set(rmse)
)

autoplot(tuned)

# finalize and fit
best_params <- select_best(tuned, metric = "rmse")
final_wf  <- finalize_workflow(wf, best_params)
final_fit <- fit(final_wf, data = train_data)

# predict and back-transform from log1p, clip at 0
test_predictions <- predict(final_fit, new_data = test_data) %>%
  mutate(.pred = pmax(0, exp(.pred) - 1))

# Create Kaggle submission format (preserve original datetime strings)
kaggle_submission <- test_predictions %>%
  bind_cols(., test_data) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

# Write to CSV file
vroom_write(x = kaggle_submission, file = "./Boost_Predictions.csv", delim = ",")
