# BikeShare Penalized Regression Analysis
# Load required libraries
library(tidyverse)
library(vroom)

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
  step_date(datetime, features = "dow") %>%
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
  #update date_time to ID
  update_role(datetime, new_role = "ID")

# Prep the recipe
prepped_recipe <- prep(my_recipe, training = train_data)

# Bake the data
baked_data <- bake(prepped_recipe, new_data = test_data)

vroom_write(x = baked_data, file = "./Baked_Test_Data.csv", delim = ",")

pred_data <- vroom("data/DataRobotResults.csv")

kaggle_submission <- pred_data %>%
  #bind_cols(., pred_data) %>%
  select(datetime, count_PREDICTION) %>%
  rename(count = count_PREDICTION) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(x = kaggle_submission, file = "./DataRobot_Solution.csv", delim = ",")
