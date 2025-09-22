# BikeShare Penalized Regression Analysis
# Load required libraries
install.packages("tidyverse")
install.packages("tidymodels")
install.packages("rpart")
install.packages("vroom")
library(tidyverse)
library(tidymodels)
library(dplyr)
library(rpart)
library(vroom)


train_data <- vroom("data/train.csv") %>%
  select(-casual, -registered)
test_data <- vroom("data/test.csv")

# ===============================================================
# DATA CLEANING
# ===============================================================

tree_recipe <- recipe(count ~ ., data = train_data) %>%
  step_log(count, offset = 1, skip = TRUE)

# ===============================================================
# GROW REGRESSION TREE
# ===============================================================
my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

# ===============================================================
# PENALIZED REGRESSION
# ===============================================================

# Establish a workflow
wf <- workflow() %>%
  add_recipe(tree_recipe) %>%
  add_model(my_mod)

# Create the parameter set automatically from the workflow and build a grid
param_set <- tune::extract_parameter_set_dials(wf)

# set grid
grid <- grid_regular(param_set, levels = 5)

# Split data for Cross Validation
folds <- vfold_cv(train_data ,v = 10 ,repeats = 1)

metrics_spec <- yardstick::metric_set(rmse, rsq)

# Run the tree
tuned <- tune_grid(
  wf,
  resamples = folds,
  grid = grid,
  metrics = metrics_spec
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
kaggle_submission <- bind_cols(test_data %>% select(datetime), test_predictions) %>%
  transmute(datetime = as.character(datetime), count = .pred)

# Write to CSV file
vroom_write(x = kaggle_submission, file = "./Reg_Tree_Predictions.csv", delim = ",")
