# Bank Marketing Campaign Analysis

## Problem Statement 

This project aims to predict whether a client will subscribe to a term deposit based on their demographic and interaction data with a Portuguese banking institution. The dataset is highly imbalanced, with a majority of clients not subscribing to the term deposit.

## Dataset Overview

The dataset includes various attributes related to clients' demographics, previous interactions, and socio-economic indicators. The target variable is binary, indicating whether the client subscribed to a term deposit.

## Data Preprocessing

### Data Cleaning
- **Missing Values**: Missing values were identified and handled.
- **Encoding**: Categorical variables were encoded using one-hot encoding.
- **Normalization**: Continuous variables were normalized to improve model performance.

### Data Balancing
SMOTE (Synthetic Minority Over-sampling Technique) was applied to handle the class imbalance by oversampling the minority class (clients who subscribed).

## Exploratory Data Analysis (EDA)

### Insights
- Older clients were more likely to subscribe.
- Clients with positive outcomes in past campaigns were more likely to subscribe.
- Clients with higher balances showed a higher likelihood of subscription.

### Visualizations
- **Target Variable Distribution**: Showed the imbalance in the dataset.
- **Correlation Matrix**: Highlighted significant relationships between features.

## Model Selection

### Baseline Model: Logistic Regression
- **Type**: Supervised learning, Linear Model
- **Description**: Implemented as a baseline to establish a performance benchmark.

### Complex Model: Random Forest
- **Type**: Supervised learning, Ensemble Learning Algorithm
- **Description**: Selected for its ability to handle large datasets and reduce overfitting by averaging multiple decision trees.

### Model Training
Both models were trained on the balanced dataset, with the Random Forest model fine-tuned for optimal performance.

## Model Evaluation

### Metrics
- **Accuracy**: Overall correctness of the model.
- **Precision**: Accuracy of positive predictions.
- **Recall**: Ability to find all positive instances.
- **F1-Score**: Balances precision and recall.
- **Confusion Matrix**: Visualizes model performance in predicting both classes.

### Results
- **Logistic Regression**:
  - Accuracy: 87.6%
  - Precision: 0.40
  - Recall: 0.52
  - F1-Score: 0.45
- **Random Forest**:
  - Accuracy: 92.1%
  - Precision: 0.58
  - Recall: 0.55
  - F1-Score: 0.56

### Model Justification
The Random Forest model was chosen due to its superior performance in accuracy, precision, and F1-Score compared to the logistic regression model.

## Feature Importance

### Key Insights
- **Previous Outcome**: Most significant feature influencing subscription decisions.
- **Duration**: Duration of the last contact was crucial in predicting client behavior.
- **Balance**: Higher account balances were strongly associated with higher subscription rates.

## Conclusion

This project highlights the effectiveness of supervised learning, particularly the Random Forest algorithm, in handling imbalanced classification problems. The insights gained from this analysis can help in optimizing future marketing strategies.

# Apple Leaf Disease Detection

##  Problem Statement
This project focuses on identifying whether an apple leaf is healthy or unhealthy using images from the PlantVillage dataset. The dataset contains images of healthy leaves and those affected by various diseases.

## Objective
To develop a binary classification model that predicts the health status of apple leaves (Healthy or Unhealthy) using image data.

## Dataset
The dataset consists of 3,164 unique images of apple leaves, split into:
- **Healthy Leaves:** 1,638 images
- **Unhealthy Leaves:** 1,526 images (comprising Cedar Apple Rust, Black Rot, and Apple Scab diseases)

## Models & Algorithms
Two supervised learning models were employed:

1. **Baseline Model:** Convolutional Neural Network (CNN)
   - **Type:** Supervised Learning
   - **Algorithm:** CNN with standard layers including Conv2D, MaxPooling2D, and Dense layers.

2. **Complex Model:** MobileNet
   - **Type:** Supervised Learning
   - **Algorithm:** MobileNet, a lightweight Convolutional Neural Network optimized for mobile and embedded vision applications.

## Data Preprocessing
- **Normalization:** All images were resized to 224x224 pixels and pixel values were normalized.
- **Data Augmentation:** Techniques such as rotation, shifting, and zooming were applied to increase dataset diversity.
- **Anomaly Detection:** Identified and removed 15 anomalous images based on brightness levels to improve model quality.

## Model Performance
### Baseline CNN Model
- **Accuracy:** 88.32%
- **Precision:** 0.4375
- **Recall:** 0.5185
- **F1-Score:** 0.4746

### MobileNet Model
- **Accuracy:** 98.19%
- **Precision:** 0.4704
- **Recall:** 0.4545
- **F1-Score:** 0.4623

## Conclusion
The MobileNet model outperformed the baseline CNN in terms of accuracy (98.19% vs 88.32%). However, the CNN model demonstrated better recall, making it more effective at detecting unhealthy leaves. The choice between these models depends on the application's priority—whether accuracy or recall is more critical.


# Knapsack Problem Optimization

## Problem Statement

This project tackles the classic Knapsack Problem, an optimization problem that involves selecting a combination of items to maximize the total value within a knapsack without exceeding its weight capacity. The problem is formulated as follows:

Given a set of items, each with a specific weight and value, and a knapsack with a maximum weight capacity, the objective is to determine the combination of items that maximizes the total value without exceeding the knapsack's weight capacity.

## Objective

To find an approximate solution to the Knapsack Problem using a Genetic Algorithm. The algorithm aims to predict whether an item should be included in the knapsack to maximize the overall value.

## Dataset Overview

The dataset used for this project consists of randomly generated values and weights for a large number of items. These values and weights are utilized to simulate the Knapsack Problem and test the performance of the Genetic Algorithm.

### Example:
- **Weights:** `[4, 3, 9, 11]`
- **Values:** `[33, 24, 100, 93]`
- **Max Knapsack Capacity:** 500

## Exploratory Data Analysis

A scatter plot is generated to visualize the relationship between item weights and values. This helps to understand the distribution and correlation between these two variables, which is crucial for the optimization process.

### Scatter Plot of Weights vs. Values
The scatter plot shows the weights on the x-axis and values on the y-axis, with each item represented as a point. The plot provides insights into how the values vary with weights, helping in understanding which items might be more valuable for the knapsack.

## Genetic Algorithm Implementation

### Solution Representation
Each potential solution (individual) is represented as a binary string (list), where each bit indicates whether an item is included in the knapsack (1) or not (0).

### Parameters and Population Initialization
- **Population Size:** 200
- **Generations:** 300
- **Mutation Rate:** 0.001
- **Tournament Size:** 5

A population of binary strings is generated, where each string represents a possible solution. The initial population is created to ensure that all solutions are feasible and do not exceed the knapsack's capacity.

### Fitness Function
The fitness function calculates the total value of items in the knapsack. If the total weight exceeds the knapsack's capacity, the fitness is penalized to zero or the solution is discarded.

### Selection, Crossover, and Mutation
- **Selection:** Tournament selection is used to choose the best individuals from a random sample for the next generation.
- **Crossover:** Single-point crossover is applied to swap segments of two parents to create offspring.
- **Mutation:** Random bits in the individual's binary string are flipped with a probability defined by the mutation rate.

### Genetic Algorithm Workflow
The algorithm initializes the population, evaluates the fitness of each individual, selects the best individuals, and applies crossover and mutation to generate new populations over several generations.

## Results

### Best Solution Found
The best solution identified by the Genetic Algorithm represents the combination of items that maximizes the total value while staying within the knapsack's weight capacity.

- **Best Fitness (Total Value):** 21917.0
- **Number of Items Selected:** 543
- **Selected Items:** A list of item indices included in the optimal solution.

### Visualizations
1. **Scatter Plot of Selected vs. Non-Selected Items:**
   - Displays the selected items in red and non-selected items in gray, helping to visualize which items were chosen by the algorithm.

2. **Bar Plot of Selected Items and Their Values:**
   - Shows the values of the selected items, providing a clear view of the contribution of each item to the total value.

## Conclusion

This project demonstrates the application of a Genetic Algorithm to solve the Knapsack Problem, highlighting the algorithm's effectiveness in finding near-optimal solutions to complex optimization problems. The results indicate the maximum value that can be achieved with the given set of items and the knapsack's capacity, along with a detailed view of the selected items.

