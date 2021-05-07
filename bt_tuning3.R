# load packages
library(tidyverse)
library(tidymodels)
library(naniar)
library(lubridate)

# set seed
set.seed(15)

# load data
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

# Define model ----

bt_model <- boost_tree(mode = "classification",
                       min_n = tune(),
                       mtry = tune(),
                       learn_rate = tune()) %>%
  set_engine("xgboost")


# set-up tuning grid ----
bt_params <- parameters(bt_model) %>%
  update(mtry = mtry(range = c(1, 30)),
         learn_rate = learn_rate(range = c(-5, -.2)))

# define tuning grid
bt_grid <- grid_regular(bt_params, levels = 5)

# workflow ----
bt_workflow <- workflow() %>%
  add_model(bt_model) %>%
  add_recipe(loan_recipe)

# Tuning/fitting ----
bt_tuned <- bt_workflow %>%
  tune_grid(loan_fold, grid = bt_grid)

# save results for fast access
save(bt_tuned, bt_workflow, file = "stat-301-3-classification-2021-loan-repayment/bank_loans_tune_bt3.rda")

# chose workflow with best accuracy
bt_workflow_tuned <- bt_workflow %>%
  finalize_workflow(select_best(bt_tuned, metric = "accuracy"))
# fit training data to the best model
bt_results <- fit(bt_workflow_tuned, l_train)

# make predictions for testing data
bt_pred <- predict(bt_results, type = "class", l_test) %>%
  bind_cols(id = l_test$id)
# organize results for kaggle
bt_pred <- bt_pred[, c(2,1)] %>%
  rename(Category = .pred_class)

# write out results
write_csv(bt_pred, "stat-301-3-classification-2021-loan-repayment/bt_pred3.csv")

