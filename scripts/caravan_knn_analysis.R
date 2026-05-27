# Caravan Insurance Purchase Prediction
# Author: Arunima Marwaha
#
# Purpose:
# This script uses K-Nearest Neighbors to predict whether customers
# purchase caravan insurance. The focus is on imbalanced classification
# and business-oriented evaluation using precision.
#
# Dataset:
# Caravan dataset from the ISLR2 package
#
# Saved outputs:
# outputs/figures/
# outputs/tables/

# -----------------------------
# 1. Setup
# -----------------------------

required_packages <- c("ISLR2", "caret", "tidyverse")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(ISLR2)
library(caret)
library(tidyverse)

# Create output folders if they do not already exist

setwd("C:/Users/ama/OneDrive/OneDrive - ROCKWOOL FONDEN/Desktop/caravan-insurance-purchase-prediction")
dir.create("outputs", showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)

# -----------------------------
# 2. Load and Inspect Data
# -----------------------------

data("Caravan")

# Save basic dataset overview
dataset_overview <- data.frame(
  metric = c("Number of observations", "Number of predictor variables", "Response variable"),
  value = c(nrow(Caravan), ncol(Caravan) - 1, "Purchase")
)

write.csv(dataset_overview,
          "outputs/tables/01_dataset_overview.csv",
          row.names = FALSE)

# Save class distribution
class_distribution <- Caravan %>%
  count(Purchase) %>%
  mutate(
    share = n / sum(n),
    percentage = round(share * 100, 2)
  )

write.csv(class_distribution,
          "outputs/tables/02_purchase_class_distribution.csv",
          row.names = FALSE)

# Plot class imbalance
p_class <- ggplot(class_distribution, aes(x = Purchase, y = n, fill = Purchase)) +
  geom_col() +
  labs(
    title = "Purchase Class Distribution",
    subtitle = "Only a small share of customers purchased caravan insurance",
    x = "Purchase",
    y = "Number of customers"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("outputs/figures/01_purchase_class_distribution.png",
       p_class, width = 7, height = 5, dpi = 300)

# -----------------------------
# 3. Preprocessing
# -----------------------------

# KNN is distance-based, so predictors must be standardized.
# The response variable Purchase is excluded before scaling.
x_standardized <- scale(Caravan[, -86])
y <- Caravan$Purchase

# Save a small check showing standardization effect
standardization_check <- data.frame(
  variable = colnames(x_standardized)[1:5],
  mean_after_scaling = apply(x_standardized[, 1:5], 2, mean),
  sd_after_scaling = apply(x_standardized[, 1:5], 2, sd)
)

write.csv(standardization_check,
          "outputs/tables/03_standardization_check.csv",
          row.names = FALSE)

# -----------------------------
# 4. Train/Test Split
# -----------------------------

# Following the exercise setup:
# observations 1-1000 are test data; remaining observations are training data.
test_index <- 1:1000

train_x <- x_standardized[-test_index, ]
test_x <- x_standardized[test_index, ]

train_y <- y[-test_index]
test_y <- y[test_index]

# Baseline: always predict "No"
baseline_error <- mean(test_y != "No")
baseline_accuracy <- 1 - baseline_error

baseline_table <- data.frame(
  model = "Baseline: Always predict No",
  error_rate = baseline_error,
  accuracy = baseline_accuracy,
  predicted_buyers = 0,
  precision_among_predicted_buyers = NA
)

write.csv(baseline_table,
          "outputs/tables/04_baseline_performance.csv",
          row.names = FALSE)

# -----------------------------
# 5. Helper Function for KNN Evaluation
# -----------------------------

evaluate_knn <- function(k_value) {
  set.seed(123)

  model <- knn3(x = train_x, y = train_y, k = k_value)
  predictions <- predict(model, test_x, type = "class")

  confusion <- table(Predicted = predictions, Actual = test_y)

  # Make sure matrix cells exist even if a class is not predicted
  predicted_yes <- ifelse("Yes" %in% rownames(confusion), sum(confusion["Yes", ]), 0)
  true_positive <- ifelse(
    "Yes" %in% rownames(confusion) & "Yes" %in% colnames(confusion),
    confusion["Yes", "Yes"],
    0
  )

  precision <- ifelse(predicted_yes == 0, NA, true_positive / predicted_yes)
  error_rate <- mean(test_y != predictions)
  accuracy <- 1 - error_rate

  performance <- data.frame(
    model = paste0("KNN: K = ", k_value),
    k = k_value,
    error_rate = error_rate,
    accuracy = accuracy,
    predicted_buyers = predicted_yes,
    true_positives = true_positive,
    precision_among_predicted_buyers = precision
  )

  confusion_df <- as.data.frame(confusion) %>%
    mutate(k = k_value)

  list(
    model = model,
    predictions = predictions,
    confusion = confusion_df,
    performance = performance
  )
}

# -----------------------------
# 6. Fit and Evaluate KNN Models
# -----------------------------

results_k1 <- evaluate_knn(1)
results_k3 <- evaluate_knn(3)
results_k5 <- evaluate_knn(5)

performance_table <- bind_rows(
  results_k1$performance,
  results_k3$performance,
  results_k5$performance
)

write.csv(performance_table,
          "outputs/tables/05_knn_performance_comparison.csv",
          row.names = FALSE)

confusion_tables <- bind_rows(
  results_k1$confusion,
  results_k3$confusion,
  results_k5$confusion
)

write.csv(confusion_tables,
          "outputs/tables/06_knn_confusion_matrices.csv",
          row.names = FALSE)

# Add baseline for chart comparison
performance_with_baseline <- bind_rows(
  baseline_table %>%
    mutate(k = NA, true_positives = NA),
  performance_table
)

write.csv(performance_with_baseline,
          "outputs/tables/07_model_performance_with_baseline.csv",
          row.names = FALSE)

# -----------------------------
# 7. Visualize Results
# -----------------------------

# Precision by K
p_precision <- performance_table %>%
  ggplot(aes(x = factor(k), y = precision_among_predicted_buyers)) +
  geom_col(fill = "steelblue") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Precision Among Predicted Buyers by K",
    subtitle = "Precision measures how many predicted buyers actually purchased",
    x = "K value",
    y = "Precision"
  ) +
  theme_minimal()

ggsave("outputs/figures/02_precision_by_k.png",
       p_precision, width = 7, height = 5, dpi = 300)

# Number of predicted buyers by K
p_predicted_buyers <- performance_table %>%
  ggplot(aes(x = factor(k), y = predicted_buyers)) +
  geom_col(fill = "coral") +
  labs(
    title = "Number of Customers Predicted as Buyers by K",
    subtitle = "Higher K makes the model more conservative",
    x = "K value",
    y = "Predicted buyers"
  ) +
  theme_minimal()

ggsave("outputs/figures/03_predicted_buyers_by_k.png",
       p_predicted_buyers, width = 7, height = 5, dpi = 300)

# Error rate comparison
p_error <- performance_with_baseline %>%
  mutate(model = factor(model, levels = model)) %>%
  ggplot(aes(x = model, y = error_rate)) +
  geom_col(fill = "grey50") +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Error Rate Comparison",
    subtitle = "Overall accuracy can be misleading with imbalanced data",
    x = "Model",
    y = "Error rate"
  ) +
  theme_minimal()

ggsave("outputs/figures/04_error_rate_comparison.png",
       p_error, width = 8, height = 5, dpi = 300)

# Confusion matrix heatmap
p_confusion <- confusion_tables %>%
  ggplot(aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  facet_wrap(~ k) +
  labs(
    title = "KNN Confusion Matrices",
    x = "Actual class",
    y = "Predicted class"
  ) +
  theme_minimal()

ggsave("outputs/figures/05_confusion_matrices.png",
       p_confusion, width = 8, height = 5, dpi = 300)

# -----------------------------
# 8. Business Interpretation
# -----------------------------

business_summary <- data.frame(
  insight = c(
    "The dataset is highly imbalanced, with only a small share of customers purchasing caravan insurance.",
    "A naive model that always predicts 'No' has a low error rate but does not identify any potential buyers.",
    "KNN models identify a small set of likely buyers, making them more useful for targeted marketing than overall accuracy alone suggests.",
    "As K increases, the model becomes more conservative and predicts fewer buyers.",
    "Precision among predicted buyers is more relevant than overall accuracy when the business goal is efficient customer targeting."
  )
)

write.csv(business_summary,
          "outputs/tables/08_business_interpretation.csv",
          row.names = FALSE)

# -----------------------------
# 9. Console Summary
# -----------------------------

cat("\nAnalysis complete.\n")
cat("Figures saved in: outputs/figures/\n")
cat("Tables saved in: outputs/tables/\n")
cat("\nPerformance summary:\n")
print(performance_table)
