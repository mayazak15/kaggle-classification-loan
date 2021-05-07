# load packages
library(tidyverse)
library(tidymodels)
library(naniar)

# set seed
set.seed(15)

# load in data
l_train <- read_csv("stat-301-3-classification-2021-loan-repayment/train.csv")
l_test <- read_csv("stat-301-3-classification-2021-loan-repayment/test.csv")

# make predictor a factor
l_train <- l_train  %>%
  mutate(hi_int_prncp_pd = as.factor(hi_int_prncp_pd))

# set up cross validaiton with repeats
loan_fold <- vfold_cv(data = l_train, v = 5, repeats = 3)

# recipe
loan_recipe <-recipe(hi_int_prncp_pd ~ term + out_prncp_inv + int_rate + application_type +
                       loan_amnt + tot_coll_amt + emp_length + annual_inc + avg_cur_bal + home_ownership + dti, data = l_train) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>% # one-hot encode all categorical predictors
  step_normalize(all_numeric()) %>%
  step_interact(hi_int_prncp_pd ~ (.)^2)

prep(loan_recipe) %>%
  bake(new_data = NULL)

# Define model ----

rf_model <- rand_forest(mode = "classification",
                       min_n = tune(),
                       mtry = tune()) %>%
  set_engine("ranger")


# set-up tuning grid ----
rf_params <- parameters(rf_model) %>%
  update(mtry = mtry(range = c(1, 30)))

# define tuning grid
rf_grid <- grid_regular(rf_params, levels = 5)

# workflow ----
rf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(loan_recipe)

# Tuning/fitting ----
rf_tuned <- rf_workflow %>%
  tune_grid(loan_fold, grid = rf_grid)
# save results for fast access
save(rf_tuned, rf_workflow, file = "stat-301-3-classification-2021-loan-repayment/bank_loans_tune_rf.rda")

# chose workflow with best accuracy
rf_workflow_tuned <- rf_workflow %>%
  finalize_workflow(select_best(rf_tuned, metric = "accuracy"))
# fit training data to the best workflow
rf_results <- fit(rf_workflow_tuned, l_train)

# make predictions for testing data
rf_pred <- predict(rf_results, type = "class", l_test) %>%
  bind_cols(id = l_test$id)
# organize results for kaggle
rf_pred <- rf_pred[, c(2,1)] %>%
  rename(Category = .pred_class)

# write ou results
write_csv(rf_pred, "stat-301-3-classification-2021-loan-repayment/rf_pred.csv")
