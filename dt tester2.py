import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score

# Step 1: Generate a synthetic dataset
X, y = make_classification(
    n_samples=1000,          # Total samples
    n_features=10,           # Number of features
    n_informative=5,         # Number of informative features
    n_redundant=2,           # Number of redundant features
    n_classes=2,             # Number of classes
    flip_y=0.3,              # Introduce noise in labels (30% mislabeled data)
    class_sep=0.5,           # Reduce class separation for overlap
    random_state=42          # Ensure reproducibility
)

# Convert to DataFrame
df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(1, 11)])
df['Target'] = y

# Save dataset to CSV (optional)
df.to_csv("synthetic_dataset.csv", index=False)
print("Synthetic dataset saved as 'synthetic_dataset.csv'.")

# Check class distribution
print("\nClass Distribution:")
print(df['Target'].value_counts())

# Step 2: Prepare data for training
X = df.drop(columns=["Target"])
y = df["Target"]

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train the Decision Tree model
dt = DecisionTreeClassifier(max_depth=4, random_state=42)  # Fine-tuned depth
dt.fit(X_train, y_train)

# Step 4: Predict and Evaluate
y_pred = dt.predict(X_test)

# Evaluate performance
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nPerformance Metrics:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Step 5: Plot Confusion Matrix
ConfusionMatrixDisplay.from_estimator(dt, X_test, y_test, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Step 6: Save the trained model
import pickle
pickle.dump(dt, open('decision_tree_model.pkl', 'wb'))
print("\nModel saved as 'decision_tree_model.pkl'.")
