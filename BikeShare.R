# BikeShare Penalized Regression Analysis
# Load required libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(parsnip)
library(lubridate)

train_data <- vroom("data/train.csv") %>%
  select(-casual, -registered)
test_data <- vroom("data/test.csv")

# ===============================================================
# DATA CLEANING
# ===============================================================


my_recipe <- recipe(count ~ ., data = train_data) %>%
  step_log(count, offset = 1, skip = TRUE) %>%
  
  # recode + preserve originals for later bins
  step_mutate(
    weather = ifelse(weather == 4, 3, weather),
    temp_orig = temp, atemp_orig = atemp, humidity_orig = humidity, windspeed_orig = windspeed
  ) %>%
  
  # time parts
  step_time(datetime, features = "hour") %>%
  step_date(datetime, features = c("dow","month","doy")) %>%
  step_mutate(
    hour_num = as.numeric(datetime_hour) - 1,
    dow_num  = as.numeric(datetime_dow),
    mon_num  = as.numeric(datetime_month),
    doy_num  = as.numeric(datetime_doy)
  ) %>%
  
  # calendar flags
  step_mutate(
    month = mon_num,
    is_weekend      = as.integer(dow_num %in% c(1,7)),
    is_morning_rush = as.integer(hour_num %in% 7:9),
    is_evening_rush = as.integer(hour_num %in% 17:19),
    time_of_day = case_when(
      hour_num %in% 6:11  ~ "morning",
      hour_num %in% 12:17 ~ "afternoon",
      hour_num %in% 18:22 ~ "evening",
      TRUE ~ "night"
    )
  ) %>%
  
  # fourier features
  step_mutate(
    hour_s1 = sin(2*pi*hour_num/24), hour_c1 = cos(2*pi*hour_num/24),
    hour_s2 = sin(4*pi*hour_num/24), hour_c2 = cos(4*pi*hour_num/24),
    hour_s3 = sin(6*pi*hour_num/24), hour_c3 = cos(6*pi*hour_num/24),
    doy_s1  = sin(2*pi*doy_num/365), doy_c1  = cos(2*pi*doy_num/365),
    doy_s2  = sin(4*pi*doy_num/365), doy_c2  = cos(4*pi*doy_num/365)
  ) %>%
  
  # time buckets
  step_mutate(
    wknd_prox    = pmin(abs(dow_num-6), abs(dow_num-7)),
    block_2h     = factor(sprintf("%02d-%02d", hour_num - hour_num%%2, hour_num - hour_num%%2 + 1)),
    hour_of_week = (dow_num-1)*24 + hour_num
  ) %>%
  
  # coarse weather buckets on originals
  step_mutate(
    wind_category = case_when(
      windspeed_orig < 10 ~ "calm",
      windspeed_orig < 20 ~ "moderate",
      TRUE ~ "windy"
    ),
    wind_bin5     = cut(windspeed_orig, breaks = c(-Inf,5,10,15,25,Inf), include.lowest = TRUE),
    humidity_bin5 = cut(humidity_orig, breaks = c(-Inf,20,40,60,80,Inf), include.lowest = TRUE),
    temp_bin6     = cut(temp_orig,     breaks = c(-Inf,0,5,10,15,20,Inf), include.lowest = TRUE)
  ) %>%
  
  # explicit interactions (numeric only; use *_orig)
  step_mutate(
    temp_humidity = temp_orig * humidity_orig,
    h1s_dow = hour_s1 * dow_num, h1c_dow = hour_c1 * dow_num,
    h2s_dow = hour_s2 * dow_num, h2c_dow = hour_c2 * dow_num,
    temp_wknd = temp_orig * wknd_prox,  hum_wknd = humidity_orig * wknd_prox,
    temp_hour = temp_orig * hour_num,   hum_hour = humidity_orig * hour_num,
    wind_morn = windspeed_orig * is_morning_rush,
    wind_eve  = windspeed_orig * is_evening_rush,
    wet_cool  = as.integer(humidity_orig >= 70 & temp_orig <= 15),
    hot_humid = as.integer(humidity_orig >= 70 & temp_orig >= 25),
    wetcool_dow  = wet_cool * dow_num,
    hothumid_dow = hot_humid * dow_num
  ) %>%
  
  # holiday features
  step_mutate(
    days_from_holiday = abs(as.numeric(as.Date(datetime) - as.Date("2011-01-01"))),
    is_special_day    = as.integer(holiday == 1 | workingday == 0)
  ) %>%
  
  # clamp before fine cuts to avoid NA bins at bake
  step_mutate(
    temp_clip      = pmin(pmax(temp_orig,     -10), 40),
    atemp_clip     = pmin(pmax(atemp_orig,    -10), 50),
    humidity_clip  = pmin(pmax(humidity_orig,   0), 100),
    windspeed_clip = pmin(pmax(windspeed_orig,  0),  40)
  ) %>%
  
  # fine bins and 2D grid
  step_mutate(
    hour_block_1h   = factor(sprintf("%02d", hour_num)),
    hour_of_week_f  = factor(hour_of_week),
    dow_f           = factor(dow_num),
    mon_f           = factor(mon_num),
    
    temp_bin12      = cut(temp_clip,      breaks = seq(-10, 40, by = 4),  include.lowest = TRUE, right = FALSE),
    atemp_bin12     = cut(atemp_clip,     breaks = seq(-10, 50, by = 5),  include.lowest = TRUE, right = FALSE),
    humidity_bin10  = cut(humidity_clip,  breaks = seq(0, 100, by = 10),  include.lowest = TRUE, right = FALSE),
    wind_bin10      = cut(windspeed_clip, breaks = c(0,2,4,6,8,10,12,14,16,18,Inf), include.lowest = TRUE, right = FALSE),
    
    temp_bin6g      = cut(temp_clip,     breaks = seq(-10, 40, by = 8),  include.lowest = TRUE, right = FALSE),
    humidity_bin5g  = cut(humidity_clip, breaks = seq(0, 100, by = 20),  include.lowest = TRUE, right = FALSE),
    temp_hum_grid   = factor(paste0(as.character(temp_bin6g), "__", as.character(humidity_bin5g)))
  ) %>%
  
  # convert main categoricals to factors
  step_mutate(
    weather       = factor(weather),
    season        = factor(season),
    time_of_day   = factor(time_of_day),
    wind_category = factor(wind_category),
    wind_bin5     = factor(wind_bin5),
    humidity_bin5 = factor(humidity_bin5),
    temp_bin6     = factor(temp_bin6)
  ) %>%
  
  # transforms on true numeric originals
  step_discretize(temp,     num_breaks = 4, min_unique = 5) %>%
  step_discretize(humidity, num_breaks = 3, min_unique = 5) %>%
  step_poly(atemp, degree = 2) %>%
  step_poly(windspeed, degree = 2) %>%
  
  # drop helpers before dummies
  step_rm(
    hour_num, dow_num, mon_num, doy_num,
    temp_bin6g, humidity_bin5g
  ) %>%
  
  # handle novel/unknown then expand dummies
  step_novel(all_nominal_predictors(), new_level = "__new__") %>%
  step_unknown(all_nominal_predictors(), new_level = "__unknown__") %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  
  # clean numerics after expansion
  step_impute_median(all_numeric_predictors()) %>%
  step_zv(all_numeric_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.95) %>%
  step_normalize(all_numeric_predictors()) %>%
  
  # roles
  update_role(datetime, new_role = "ID")

#set up BART Model
bart_model <- bart(trees = tune()) %>%
  set_engine("dbarts") %>%
  set_mode("regression")

#Establish a workflow
wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model)

#set grid
grid <- grid_regular(trees() ,levels = 15)

# Split data for Cross Validation
folds <- vfold_cv(train_data ,v = 4 ,repeats = 1)

# grow the forest
tuned <- tune_grid(
  wf,
  resamples = folds,
  grid = grid,
  metrics = metric_set(rmse)
)

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


vroom_write(x = kaggle_submission, file = "./BART_Solution.csv", delim = ",")
