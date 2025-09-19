# =============================================================================
# ORIGINAL LINEAR REGRESSION MODEL (NO DATA WRANGLING)
# =============================================================================

cat("\n=== FITTING ORIGINAL LINEAR REGRESSION MODEL ===\n")

# Create original training data (without log transformation for this model)
original_train_data <- vroom("data/train.csv") %>%
  select(-casual, -registered)

# Fit original linear regression model
original_linear_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression") %>%
  fit(formula = count ~ . - datetime, data = original_train_data)

# Make predictions
original_predictions <- predict(original_linear_model, new_data = test_data)

# Create Kaggle submission format
original_kaggle_submission <- original_predictions %>%
  bind_cols(., test_data) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

# Write to CSV file
vroom_write(x = original_kaggle_submission, file = "./OriginalLinear_Predictions.csv", delim = ",")
cat("Original Linear Regression predictions saved to: ./OriginalLinear_Predictions.csv\n")

# =============================================================================
# WORKFLOW-BASED LINEAR REGRESSION MODEL (WITH DATA WRANGLING)
# =============================================================================
# This model uses the data wrangling recipe from BikeShareDataWrangling.R

cat("\n=== FITTING WORKFLOW-BASED LINEAR REGRESSION MODEL ===\n")

# Create linear regression model specification
workflow_linear_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

# Create workflow
workflow_linear_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(workflow_linear_model)

# Fit the workflow
workflow_linear_fit <- workflow_linear_workflow %>%
  fit(data = train_data)

# Make predictions
workflow_predictions <- workflow_linear_fit %>%
  predict(new_data = test_data)

# Backtransform the log(count) predictions
workflow_predictions <- workflow_predictions %>%
  mutate(.pred = exp(.pred) - 1) %>%
  mutate(.pred = pmax(0, .pred))

# Create Kaggle submission format
workflow_kaggle_submission <- workflow_predictions %>%
  bind_cols(., test_data) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

# Write to CSV file
vroom_write(x = workflow_kaggle_submission, file = "./WorkflowLinear_Predictions.csv", delim = ",")
cat("Workflow Linear Regression predictions saved to: ./WorkflowLinear_Predictions.csv\n")

# =============================================================================
# PENALIZED REGRESSION MODELS (WITH DATA WRANGLING)
# =============================================================================
# This section fits multiple penalized regression models with different combinations
# of penalty and mixture parameters:

cat("\n=== FITTING PENALIZED REGRESSION MODELS ===\n")

# Define different penalty and mixture combinations
penalty_mixture_combinations <- tibble(
  model_name = c("Ridge_Low", "Ridge_High", "Lasso_Low", "Lasso_High", "ElasticNet_1", "ElasticNet_2"),
  penalty = c(0.01, 0.1, 0.01, 0.1, 0.05, 0.2),
  mixture = c(0, 0, 1, 1, 0.5, 0.3)
)

cat("Penalty and mixture combinations:\n")
print(penalty_mixture_combinations)
cat("\n")

# Create a list to store all fitted models
fitted_models <- list()

# Fit penalized regression models for each combination
for(i in 1:nrow(penalty_mixture_combinations)) {
  # Extract parameters for current model
  current_penalty <- penalty_mixture_combinations$penalty[i]
  current_mixture <- penalty_mixture_combinations$mixture[i]
  current_name <- penalty_mixture_combinations$model_name[i]
  
  cat("\n--- Fitting", current_name, "Model ---\n")
  cat("Penalty (lambda):", current_penalty, "\n")
  cat("Mixture (alpha):", current_mixture, "\n")
  
  # Create penalized regression model specification
  penalized_model <- linear_reg(
    penalty = current_penalty,
    mixture = current_mixture
  ) %>%
    set_engine("glmnet") %>%
    set_mode("regression")
  
  # Create workflow
  penalized_workflow <- workflow() %>%
    add_recipe(my_recipe) %>%
    add_model(penalized_model)
  
  # Fit the model
  fitted_model <- penalized_workflow %>%
    fit(data = train_data)
  
  # Store the fitted model
  fitted_models[[current_name]] <- fitted_model
  
  # Make predictions on test data
  test_predictions <- fitted_model %>%
    predict(new_data = test_data)
  
  # Backtransform the log(count) predictions
    test_predictions <- test_predictions %>%
    mutate(.pred = exp(.pred) - 1) %>%
    # Ensure predictions are non-negative (bike counts can't be negative)
    mutate(.pred = pmax(0, .pred))
  
  # Create Kaggle submission format
  kaggle_submission <- test_predictions %>%
    bind_cols(., test_data) %>%
    select(datetime, .pred) %>%
    rename(count = .pred) %>%
    mutate(count = pmax(0, count)) %>%
    mutate(datetime = as.character(format(datetime)))
  
  # Write to CSV file with model name
  filename <- paste0("./", current_name, "_Predictions.csv")
  vroom_write(x = kaggle_submission, file = filename, delim = ",")
  
  cat("Predictions saved to:", filename, "\n")
  cat("Sample predictions:", head(test_predictions$.pred, 3), "\n")
}

print(penalty_mixture_combinations)