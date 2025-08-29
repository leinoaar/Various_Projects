# Recipe Site Traffic Analysis

This project analyzes recipe site traffic data and builds predictive models to understand what factors contribute to **high traffic recipes**. The workflow includes data cleaning, exploratory analysis, and model development using R.

## Project Workflow

### 1. Data Validation & Cleaning
- Loaded `recipe_site_traffic_2212.csv`
- Checked structure, duplicates, and missing values
- Converted `high_traffic` into a binary variable (0/1)
- Cleaned `servings` column (removed non-numeric characters)
- Encoded categorical variables (`category`)
- Removed missing values and unnecessary columns
- Checked outliers in calories, protein, carbohydrate, and sugar

### 2. Exploratory Data Analysis
- Histograms for calorie distribution
- Bar charts for recipe categories
- Scatter plots (e.g., calories vs. protein)
- Boxplots comparing calories across high vs. low traffic recipes

### 3. Model Development
- Split data into training (80%) and testing (20%)
- Scaled numerical variables
- Models used:
  - **Logistic Regression** (baseline)
  - **Random Forest** (comparison model)

### 4. Model Evaluation
- Confusion matrices and precision scores
- Accuracy metrics
- ROC curves and AUC for both models
- Feature importance from Random Forest

## Tools & Libraries
- R base packages (`readr`, `caret`)
- **Visualization**: `ggplot2`
- **Models**: `glm` (logistic regression), `randomForest`
- **Evaluation**: `pROC`

## Results
- Logistic regression provided a baseline model.
- Random Forest achieved higher accuracy and AUC, and identified important predictors such as calories, protein, and category.
- Visualizations and ROC curves highlight differences in model performance.
