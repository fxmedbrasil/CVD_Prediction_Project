# Load necessary libraries
library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(corrplot)

# Define URLs for the datasets
diabetes_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv"
heart_failure_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"

# Define local file paths for saving
diabetes_csv <- "diabetes_data_upload.csv"
heart_failure_csv <- "heart_failure_clinical_records_dataset.csv"

# Download the datasets
download.file(diabetes_url, diabetes_csv, mode = "wb")
download.file(heart_failure_url, heart_failure_csv, mode = "wb")

# Load the datasets
diabetes_data <- read_csv(diabetes_csv)
heart_failure_data <- read_csv(heart_failure_csv)

# Data Cleaning: Handle missing values and convert categorical variables in diabetes dataset
diabetes_data <- diabetes_data %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .))) %>%
  mutate(Gender = ifelse(Gender == "Male", 1, 0))

# Rename columns with spaces in diabetes_data
colnames(diabetes_data) <- make.names(colnames(diabetes_data))

# Convert "Yes"/"No" to 1/0 for relevant columns
binary_columns <- c("Polyuria", "Polydipsia", "sudden.weight.loss", "weakness", 
                    "Polyphagia", "Genital.thrush", "visual.blurring", "Itching", 
                    "Irritability", "delayed.healing", "partial.paresis", 
                    "muscle.stiffness", "Alopecia", "Obesity", "class")
diabetes_data[binary_columns] <- lapply(diabetes_data[binary_columns], function(x) ifelse(x == "Yes", 1, 0))
diabetes_data$class <- as.factor(diabetes_data$class)

# Split the diabetes data into training and test sets
set.seed(123)
diabetes_train_index <- createDataPartition(diabetes_data$class, p = 0.8, list = FALSE)
diabetes_train_data <- diabetes_data[diabetes_train_index, ]
diabetes_test_data <- diabetes_data[-diabetes_train_index, ]

# Load and preprocess the heart failure dataset
heart_failure_data$DEATH_EVENT <- as.factor(heart_failure_data$DEATH_EVENT)

# Define common parameters that link diabetes and heart failure (focusing on cardiovascular and metabolic health)
common_params <- c("age", "serum_creatinine", "serum_sodium", "ejection_fraction", "high_blood_pressure", "diabetes", "DEATH_EVENT")

# Data Splitting - Heart Failure Data
set.seed(123)
heart_failure_train_index <- createDataPartition(heart_failure_data$DEATH_EVENT, p = 0.8, list = FALSE)
heart_failure_train <- heart_failure_data[heart_failure_train_index, ]
heart_failure_test <- heart_failure_data[-heart_failure_train_index, ]

# Check for class balance in training data
print("Class distribution in the training data:")
print(table(heart_failure_train$DEATH_EVENT))

# Downsampling the majority class if there is class imbalance
if (length(unique(heart_failure_train$DEATH_EVENT)) >= 2) {
  minority_class <- heart_failure_train %>% filter(DEATH_EVENT == 1)
  majority_class <- heart_failure_train %>% filter(DEATH_EVENT == 0)
  
  if (nrow(minority_class) > 0 && nrow(majority_class) > 0) {
    set.seed(123)
    majority_class_downsampled <- majority_class[sample(nrow(majority_class), nrow(minority_class)), ]
    heart_failure_train <- rbind(majority_class_downsampled, minority_class)
    heart_failure_train$DEATH_EVENT <- as.factor(heart_failure_train$DEATH_EVENT)
    
    print("Class distribution after downsampling:")
    print(table(heart_failure_train$DEATH_EVENT))
  } else {
    stop("The training data does not contain enough data to perform downsampling.")
  }
} else {
  stop("The training data does not contain at least two classes. Repartitioning is required.")
}

# Baseline Model: Logistic Regression on Heart Failure Data
logit_model <- glm(DEATH_EVENT ~ ., data = heart_failure_train[, common_params], family = binomial)
logit_predictions <- predict(logit_model, newdata = heart_failure_test[, common_params], type = "response")
logit_predicted_classes <- ifelse(logit_predictions > 0.5, 1, 0)
logit_predicted_factors <- as.factor(logit_predicted_classes)
logit_conf_matrix <- confusionMatrix(logit_predicted_factors, heart_failure_test$DEATH_EVENT)
print("Confusion Matrix - Logistic Regression:")
print(logit_conf_matrix)

# Advanced Model: Random Forest on Heart Failure Data
rf_model <- randomForest(DEATH_EVENT ~ ., data = heart_failure_train[, common_params], importance = TRUE, ntree = 500)
rf_predictions <- predict(rf_model, newdata = heart_failure_test[, common_params])
rf_conf_matrix <- confusionMatrix(rf_predictions, heart_failure_test$DEATH_EVENT)
print("Confusion Matrix - Random Forest:")
print(rf_conf_matrix)

# Feature Importance from the Random Forest model
importance <- importance(rf_model)
print("Feature Importance - Random Forest:")
print(importance)
varImpPlot(rf_model)

# ROC Curve and AUC for Random Forest model
rf_probs <- predict(rf_model, newdata = heart_failure_test[, common_params], type = "prob")[,2]
roc_curve <- roc(heart_failure_test$DEATH_EVENT, rf_probs)
plot(roc_curve, col = "blue", main = "ROC Curve - Random Forest")
auc_value <- auc(roc_curve)
print(paste("AUC - Random Forest:", auc_value))
