library(tidyverse)
library(tidymodels)
library(naniar)

set.seed(15)

l_train <- read.csv("stat-301-3-classification-2021-loan-repayment/train.csv")
l_test <- read.csv("stat-301-3-classification-2021-loan-repayment/test.csv")

miss_var_table(l_train)

skimr::skim_without_charts(l_train)


l_train <- l_train  %>%
  mutate(hi_int_prncp_pd = as.factor(hi_int_prncp_pd))

# set up cross validaiton with repeats
loan_fold <- vfold_cv(data = l_train, v = 5, repeats = 3)

# recipe
loan_recipe <- recipe(hi_int_prncp_pd ~ grade + loan_amnt +  mort_acc + purpose + term, data = l_train) %>%
  # step_rm(id) %>%
  #step_zv(emp_title) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>% # one-hot encode all categorical predictors
  step_normalize(all_numeric()) %>%
  step_interact(hi_int_prncp_pd ~ (.)^2)

prep(loan_recipe) %>%
  bake(new_data = NULL)

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

save(bt_tuned, bt_workflow, file = "stat-301-3-classification-2021-loan-repayment/bank_loans_tune_bt.rda")
bt_workflow_tuned <- bt_workflow %>%
  finalize_workflow(select_best(bt_tuned, metric = "accuracy"))
bt_results <- fit(bt_workflow_tuned, l_train)


bt_pred <- predict(bt_results, type = "class", l_test) %>%
  bind_cols(id = l_test$id)

bt_pred <- bt_pred[, c(2,1)] %>%
  rename(Category = .pred_class)

write_csv(bt_pred, "stat-301-3-classification-2021-loan-repayment/bt_pred.csv")
