# PART ONE : DATA VALIDATION

# Load necessary libraries
library(readr)

# Load the dataset
data <- read.csv("recipe_site_traffic_2212.csv")

# Check the structure of the data 
str(data)
head(data)

# Check unique values in 'high_traffic' column
unique(data$high_traffic)
unique(data$servings)
# Get summary statistics for the dataset
summary(data)

# Check for missing values in specific columns
colSums(is.na(data))

# Check for duplicates
sum(duplicated(data))

# Handle 'high_traffic' column: Convert to 0 (no high traffic) and 1 (high traffic) for easier modeling
data$high_traffic <- ifelse(is.na(data$high_traffic), 0, 1)

# Correct data types for specific columns
data$category <- factor(data$category)  

# Clean servings column and convert to numeric
data$servings <- gsub("[^0-9]", "", data$servings)  # Remove non-numeric characters
data$servings <- as.numeric(data$servings) 

# Remove rows with missing values and remove the recipe number column as well
data <- data[, !(colnames(data) == "recipe")]
data_clean <- na.omit(data)

# Summary of the cleaned dataset
summary(data_clean)

# Check how many rows were removed
nrow(data) - nrow(data_clean)

# Check for outliers in numeric columns 
boxplot(data$calories, main="Calories", ylab="Calories")
boxplot(data$protein, main="Protein", ylab="Protein")
boxplot(data$carbohydrate, main="Carbohydrate", ylab="Carbohydrates")
boxplot(data$sugar, main="Sugar", ylab="Sugar")

# Check if there are any remaining missing values after cleaning
colSums(is.na(data_clean))


# Final check of cleaned data structure
str(data_clean)
head(data_clean)

# PART TWO : Exploratory Analysis

# Load libraries
library(ggplot2)

# Data visualization
# Histogram for Calories
ggplot(data_clean, aes(x = calories)) + 
  geom_histogram(binwidth = 50, fill = "lightblue", color = "black") + 
  labs(title = "Distribution of Calories", x = "Calories", y = "Frequency")

# Bar chart for Recipe Categories
ggplot(data_clean, aes(x = category)) + 
  geom_bar(fill = "lightgreen", color = "black") + 
  labs(title = "Recipe Categories", x = "Category", y = "Count") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Scatter plot for Calories and Protein
ggplot(data_clean, aes(x = sqrt(calories), y = protein)) +
  geom_point(color = "blue") +
  labs(title = "Square Root of Calories vs. Protein", x = "Square Root of Calories", y = "Protein")

# Boxplot for Calories vs. High Traffic
ggplot(data_clean, aes(x = factor(high_traffic), y = calories)) +
  geom_boxplot(fill = "lightblue", color = "black") +
  labs(title = "Calories vs. High Traffic", x = "High Traffic", y = "Calories")

# PART THREE : Model development

# Load libraries
library(caret)
install.packages("randomForest")
library(randomForest)
# Set seed for reproducibility
set.seed(2025)

# Split the dataset into train and test sets
train_ind <- createDataPartition(data_clean$high_traffic, p = 0.8, list = FALSE)
train_data <- data_clean[train_ind, ]
test_data <- data_clean[-train_ind, ]

# Then scale numerical variables
train_data$scaled_calories <- scale(train_data$calories)
train_data$scaled_protein <- scale(train_data$protein)
train_data$scaled_carbohydrate <- scale(train_data$carbohydrate)
train_data$scaled_sugar <- scale(train_data$sugar)
train_data$scaled_servings <- scale(train_data$servings)

test_data$scaled_calories <- scale(test_data$calories)
test_data$scaled_protein <- scale(test_data$protein)
test_data$scaled_carbohydrate <- scale(test_data$carbohydrate)
test_data$scaled_sugar <- scale(test_data$sugar)
test_data$scaled_servings <- scale(test_data$servings)

# Convert target variable to a factor for binary classification
train_data$high_traffic <- as.factor(train_data$high_traffic)
test_data$high_traffic <- as.factor(test_data$high_traffic)

# First fit the baseline model
base_logit_model <- glm(high_traffic ~ scaled_calories + scaled_protein + scaled_carbohydrate + scaled_sugar + scaled_servings + category, data = train_data, family = binomial)

# Check the summary
summary(base_logit_model)

# Then fit comparison model. I will use RF to have little different type of model

rf_model <- randomForest(high_traffic ~ scaled_calories + scaled_protein + scaled_carbohydrate + scaled_sugar + scaled_servings + category, 
                         data = train_data, 
                         ntree = 100,        # Number of trees
                         mtry = 3,           # Number of variables randomly sampled at each split
                         importance = TRUE)

# Check the summary
summary(rf_model)

# PART FOUR : Model Evaluation

# Load libraries
library(pROC)

# First baseline models prediction
baseline_preds <- predict(base_logit_model, test_data, type = "response")
baseline_class_preds <- ifelse(baseline_preds > 0.5, 1, 0)                       # Convert probabilities to 1 if 0.5 < or 0 if not.

# The RF prediction
rf_class_preds <- predict(rf_model, test_data, type = "response")
rf_probs <- predict(rf_model, test_data, type = "prob")

# Next confusion matrix to see how well models performed
logit_cm <- confusionMatrix(as.factor(baseline_class_preds), test_data$high_traffic)
print(logit_cm)
rf_cm <- confusionMatrix(rf_class_preds, test_data$high_traffic)
print(rf_cm)

#Calculate the precicion for baseline and rf model
TP_logit <- logit_cm$table[2, 2]
FP_logit <- logit_cm$table[1, 2]
precision_logit <- TP_logit / (TP_logit + FP_logit)
cat("Logistic Regression Precision: ", precision_logit, "\n")

TP_rf <- rf_cm$table[2, 2]       #True positives
FP_rf <- rf_cm$table[1, 2]       #False positives
precision_rf <- TP_rf / (TP_rf + FP_rf)
cat("Random Forest Precision: ", precision_rf, "\n")

# Then print the accuracy of both models
logit_metrics <- logit_cm$overall
rf_metrics <- rf_cm$overall
cat("Logistic Regression - Accuracy: ", logit_metrics["Accuracy"], "\n")
cat("RF - Accuracy: ", rf_metrics["Accuracy"], "\n")

# Also Calculate and visualize roc curve
logit_roc <- roc(test_data$high_traffic, baseline_preds)  # First for baseline model
plot(logit_roc, main = "ROC Curve for Logistic Regression")
logit_auc <- auc(logit_roc)
cat("Logistic Regression AUC: ", logit_auc, "\n")

# Then with RF model
rf_roc <- roc(test_data$high_traffic, rf_probs[, 2])  # Use the probabilities for class 1 
plot(rf_roc, main = "ROC Curve for Random Forest")
rf_auc <- auc(rf_roc)
cat("Random Forest AUC: ", rf_auc, "\n")

# Lastly print comparison dataframe
comparison_df <- data.frame(
  Model = c("Logistic Regression", "SVM"),
  Accuracy = c(logit_metrics["Accuracy"], rf_metrics["Accuracy"]),
  AUC = c(logit_auc, rf_auc)
)
comparison_df

importance(rf_model)
varImpPlot(rf_model) 
