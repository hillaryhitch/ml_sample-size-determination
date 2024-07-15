

#Bootstrap sampling is a statistical technique used to estimate the distribution of a dataset by resampling with replacement. 
#This means that given an original dataset, you create multiple new samples (called bootstrap samples) by randomly selecting data points from the original dataset, 
#allowing for the same data point to be selected more than once in each new sample. Bootstrap sampling is commonly used to estimate the accuracy of a sample statistic (like the mean, 
#median, or model accuracy) and to assess the variability of these estimates.


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 10)  # 1000 samples, 10 features
y = np.random.randint(0, 2, 1000)  # Binary target variable

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize lists to store results
train_mean_accuracies = []
val_mean_accuracies = []
train_std_devs = []
val_std_devs = []
sample_sizes = range(100, len(X_train), 100)

# Perform bootstrapping with increasing sample sizes
for m in sample_sizes:
    train_scores = []
    val_scores = []
    
    for _ in range(100):  # Number of bootstrap samples
        X_boot, y_boot = resample(X_train[:m], y_train[:m])
        
        # Train the model
        model = LogisticRegression()
        model.fit(X_boot, y_boot)
        
        # Evaluate on the training set
        y_train_predict = model.predict(X_boot)
        train_scores.append(accuracy_score(y_boot, y_train_predict))
        
        # Evaluate on the validation set
        y_val_predict = model.predict(X_val)
        val_scores.append(accuracy_score(y_val, y_val_predict))
    
    # Calculate mean and standard deviation of bootstrap scores
    train_mean_accuracies.append(np.mean(train_scores))
    train_std_devs.append(np.std(train_scores))
    val_mean_accuracies.append(np.mean(val_scores))
    val_std_devs.append(np.std(val_scores))

# Plot results
plt.errorbar(sample_sizes, train_mean_accuracies, yerr=train_std_devs, fmt='-o', capsize=5, label='Training Accuracy')
plt.errorbar(sample_sizes, val_mean_accuracies, yerr=val_std_devs, fmt='-o', capsize=5, label='Validation Accuracy')
plt.xlabel('Sample Size')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy with Increasing Sample Size')
plt.legend()
plt.show()
