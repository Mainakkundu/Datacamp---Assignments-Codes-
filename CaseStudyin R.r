#CASE STUDIES IN R
----------------------------------------------
#> head(cars2018)
# A tibble: 6 x 15
  Model           `Model Index` Displacement Cylinders Gears Transmission   MPG
  <chr>                   <int>        <dbl>     <int> <int> <chr>        <int>
1 Acura NSX                  57         3.50         6     9 Manual          21
2 ALFA ROMEO 4C             410         1.80         4     6 Manual          28
3 Audi R8 AWD                65         5.20        10     7 Manual          17
4 Audi R8 RWD                71         5.20        10     7 Manual          18
5 Audi R8 Spyder~            66         5.20        10     7 Manual          17
6 Audi R8 Spyder~            72         5.20        10     7 Manual          18
# ... with 8 more variables: Aspiration <chr>, `Lockup Torque Converter` <chr>,
#   Drive <chr>, `Max Ethanol` <int>, `Recommended Fuel` <chr>, `Intake Valves
#   Per Cyl` <int>, `Exhaust Valves Per Cyl` <int>, `Fuel injection` <chr>
----------------------------------------------------

# Print the cars2018 object
cars2018

# Plot the histogram
ggplot(cars2018, aes(x = MPG)) +
    geom_histogram(bins = 25) +
    labs(y = "Number of cars",
         x = "Fuel efficiency (mpg)")

#---- CARET---
# Load caret
library(caret)

# Split the data into training and test sets
set.seed(1234)
in_train <- createDataPartition(cars_vars$Transmission, p =0.8, list = FALSE)
training <- cars_vars[in_train, ]
testing <- cars_vars[-in_train, ]

# Load caret
library(caret)

# Split the data into training and test sets
set.seed(1234)
in_train <- createDataPartition(cars_vars$Transmission, p =0.8, list = FALSE)
training <- cars_vars[in_train, ]
testing <- cars_vars[-in_train, ]

# Train a random forest model
fit_rf <- train(log(MPG) ~ ., method = 'rf', data = training,
                trControl = trainControl(method = "none"))

# Print the model object
fit_rf

# Load caret
library(caret)

# Train a linear regression model
fit_lm <- train(log(MPG) ~ ., method = "lm", data = training,
                trControl = trainControl(method = "none"))

# Print the model object
fit_lm

# Load caret
library(caret)

# Split the data into training and test sets
set.seed(1234)
in_train <- createDataPartition(cars_vars$Transmission, p =0.8, list = FALSE)
training <- cars_vars[in_train, ]
testing <- cars_vars[-in_train, ]

# A tibble: 1 x 2
#   rmse   rsq
 # <dbl> <dbl>
#  20.5 0.879
--------------------------------------------------------------
###################################
#BOOTSTRAP SAMPLING 
###################################
# Fit the models with bootstrap resampling
cars_lm_bt <- train(log(MPG) ~ ., method = "lm", data = training,
                   trControl = trainControl(method = "boot"))
cars_rf_bt <- train(log(MPG) ~ ., method = "rf", data = training,
                   trControl = trainControl(method = "boot"))
                   
# Quick look at the models
cars_lm_bt
cars_rf_bt

'''
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 93, 93, 93, 93, 93, 93, ... 
Resampling results:

  RMSE        Rsquared   MAE       
  0.08705698  0.8511491  0.06892683

Tuning parameter 'intercept' was held constant at a value of TRUE
> cars_rf_bt
Random Forest 

93 samples
12 predictors

No pre-processing
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 93, 93, 93, 93, 93, 93, ... 
Resampling results across tuning parameters:

  mtry  RMSE        Rsquared   MAE       
   2    0.11550906  0.8079840  0.08733088
   9    0.09329373  0.8360137  0.07162074
  16    0.09582296  0.8228786  0.07371557

RMSE was used to select the optimal model using the smallest value.
The final value used for the model was mtry = 9.
'''
#-------------------------------------------

results <- testing %>%
    mutate(`Linear regression` = predict(cars_lm_bt, testing),
           `Random forest` = predict(cars_rf_bt, testing))

metrics(results, truth = MPG, estimate = `Linear regression`)
metrics(results, truth = MPG, estimate = `Random forest`)

####################
#VISUALIZE THE MODEL
####################
results %>%
    gather(Method, Result, `Linear regression`:`Random forest`) %>%   ### from TidyVerse package 
    ggplot(aes(log(MPG), Result, color = Method)) +
    geom_point(size = 1.5, alpha = 0.5) +
    facet_wrap(~Method) +
    geom_abline(lty = 2, color = "gray50") +
    geom_smooth(method = "lm")
------------------------------------------------------------
#STACKOVERFLOW CASE STUDY WHO WILL BE REMOTE DEVELOPER 
------------------------------------------------------------

# Print stackoverflow
stackoverflow

# First count for Remote
stackoverflow %>% 
    count(Remote, sort = TRUE)

# then count for Country
stackoverflow %>% 
    count(Country, sort = TRUE)
	
	
ggplot(stackoverflow, aes(Remote, YearsCodedJob)) +
    geom_boxplot() +
    labs(x = NULL,
         y = "Years of professional coding experience") 

'''
Recall that when you use the pipe operator %>% with a function like glm() 
(whose first argument is not data), 
you must specify data = . to indicate that you are piping in the modeling data set.
'''
# Build a simple logistic regression model
simple_glm <- stackoverflow %>%
        select(-Respondent) %>%
        glm(Remote ~ .,
            family = "binomial",
            data = .)

# Print the summary of the model
summary(simple_glm)

###########################
#MODEL BUILDING
##########################
# Load caret
library(caret)

stack_select <- stackoverflow %>%
    select(-Respondent)

# Split the data into training and testing sets
set.seed(1234)
in_train <- createDataPartition(stack_select$Remote,p = 0.8, list = FALSE)
training <- stack_select[in_train,]
testing <- stack_select[-in_train,]

##############
#UPSAMPLING
##############
up_train <- upSample(x = select(training, -Remote),
                     y = training$Remote,
                     yname = "Remote") %>%
    as_tibble()

up_train %>%
    count(Remote)
	
# Build a logistic regression model
stack_glm <- train(Remote ~ ., method = "glm", family = "binomial",
                   data = training,
                   trControl = trainControl(method = "boot",
                                            sampling = "up"))

# Print the model object
stack_glm

# Build a random forest model
stack_rf <- train(Remote ~ .,method='rf', 
                  data = training,
                  trControl = trainControl(method = "boot",
                                           sampling='up'))
										   

# Confusion matrix for logistic regression model
confusionMatrix(predict(stack_glm,testing),
                testing$Remote)
				
# Load yardstick
library(yardstick)

# Predict values
testing_results <- testing %>%
    mutate(`Logistic regression` = predict(stack_glm, testing),
           `Random forest` = predict(stack_rf, testing))

## Calculate accuracy
accuracy(testing_results, truth = Remote, estimate = `Logistic regression`)
accuracy(testing_results, truth = Remote, estimate = `Random forest`)

## Calculate positive predict value
ppv(testing_results, truth = Remote, estimate = `Logistic regression`)
ppv(testing_results, truth = Remote, estimate = `Random forest`)

'''
> ## Calculate accuracy
> accuracy(testing_results, truth = Remote, estimate = `Logistic regression`)
[1] 0.6542591
> accuracy(testing_results, truth = Remote, estimate = `Random forest`)
[1] 0.8747316
> 
> ## Calculate positive predict value
> ppv(testing_results, truth = Remote, estimate = `Logistic regression`)
[1] 0.1804511
> ppv(testing_results, truth = Remote, estimate = `Random forest`)
[1] 0.2333333
> 
'''
########################
#VOTING CASE STUDY
########################

# How do the reponses on the survey vary with voting behavior?
voters %>%
    group_by(turnout16_2016) %>%
    summarise(`Elections dont matter` = mean(RIGGED_SYSTEM_1_2016 <= 2),
              `Economy is getting better` = mean(econtrend_2016 == 1),
              `Crime is very important` = mean(imiss_a_2016 == 2))
			  
 Visualize difference by voter turnout
voters %>%
    ggplot(aes(econtrend_2016, ..density.., fill = turnout16_2016)) +
    geom_histogram(alpha = 0.5, position = "identity", binwidth = 1) +
    labs(title = "Overall, is the economy getting better or worse?")
	
#------- Model Validation ------
# Logistic regression
vote_glm <- train(turnout16_2016 ~ ., method = "glm", family = "binomial",
                  data = training,
                  trControl = trainControl(method = "repeatedcv",
                                           repeats = 2,
                                           sampling = "up"))

# Print vote_glm
vote_glm

# Random forest
vote_rf <- train(turnout16_2016 ~ ., method = "rf", 
                 data = training,
                 trControl = trainControl(___,
                                          ___,
                                          sampling = "up"))

# Print vote_rf
vote_rf
#################
#NUNS CASE STUDY
#################

# Load tidyverse
library(tidyverse)


# View sisters67
glimpse(sisters67)

# Plot the histogram
ggplot(sisters67, aes(x = age)) +
    geom_histogram(binwidth = 10)
	
#-------- Tidy the data (means each age wise every columns into single column (key,value using gather function) 
# Print the structure of sisters67
glimpse(sisters67)

# Tidy the data set
tidy_sisters <- sisters67 %>%
    select(-sister) %>%
    gather(key, value, -age)

# Print the structure of tidy_sisters
glimpse(tidy_sisters)

# Overall agreement with all questions varied by age
tidy_sisters %>%
    group_by(age) %>%
    summarize(value = mean(value, na.rm = TRUE))

# Number of respondents agreed or disagreed overall
tidy_sisters %>%
    count(value)
	
#----- Trying to understand how each questions being related to Age -----#
# Visualize agreement with age
tidy_sisters %>%
    filter(key %in% paste0("v", 153:170)) %>%
    group_by(key, value) %>%
    summarise(age = mean(age, na.rm = TRUE)) %>%
    ggplot(aes(value, age, color = key)) +
    geom_line(show.legend = FALSE) +
    facet_wrap(~key, nrow = 3)
	
# Remove the sister column
sisters_select <- sisters67 %>% 
    select(-sister)

# Build a simple linear regression model
simple_lm <- lm(age~., 
                data = sisters_select)

# Print the summary of the model
summary(simple_lm)

# Split the data into training and validation/test sets
set.seed(1234)
in_train <- createDataPartition(sisters_select$age, 
                                p = 0.6, list = FALSE)
training <- sisters_select[in_train, ]
validation_test <- sisters_select[-in_train, ]

# Split the validation and test sets
set.seed(1234)
in_test <- createDataPartition(validation_test$age, 
                               p = 0.5, list = FALSE)
testing <- validation_test[in_test, ]
validation <- validation_test[-in_test, ]

# Load caret
library(caret)

# Fit a CART model
sisters_cart <- train(age ~ ., method = "rpart", data = training)

# Print the CART model
sisters_cart

# Make predictions on the three models
modeling_results <- validation %>%
    mutate(CART = predict(sisters_cart,validation),
           XGB = predict(sisters_xgb, validation),
           GBM = predict(sisters_gbm, validation))

# View the predictions
modeling_results %>% 
    select(CART, XGB, GBM)
	
# Load yardstick
library(yardstick)

# Compare performace
metrics(modeling_results, truth = age, estimate = CART)
metrics(modeling_results, truth = age, estimate = XGB)
metrics(modeling_results, truth = age, estimate = GBM)

# Calculate RMSE
testing %>%
    mutate(prediction = predict(sisters_gbm, testing)) %>%
    rmse(truth = age, estimate = prediction)




