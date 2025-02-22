import matplotlib.pyplot as plt
import numpy as np

# Swapped performance metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
decision_tree_scores = [63.67, 0.66, 0.56, 0.61]  # Originally KNN metrics
knn_scores = [97.13, 0.97, 0.97, 0.97]           # Originally Decision Tree metrics

# Convert percentages to fractions for plotting
decision_tree_scores[0] = decision_tree_scores[0] / 100  # Accuracy in fraction
knn_scores[0] = knn_scores[0] / 100  # Accuracy in fraction

# Bar width and positions
bar_width = 0.35
x_positions = np.arange(len(metrics))

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(x_positions - bar_width / 2, decision_tree_scores, bar_width, label='Decision Tree', color='skyblue')
plt.bar(x_positions + bar_width / 2, knn_scores, bar_width, label='KNN', color='orange')

# Add labels, title, and legend
plt.xticks(x_positions, metrics)
plt.ylabel('Scores')
plt.ylim(0, 1.2)  # Adjust range to include percentages converted to fractions
plt.title('Performance Comparison: Decision Tree vs KNN (Swapped Metrics)')
plt.legend()

# Display values on bars
for i, (dt_score, knn_score) in enumerate(zip(decision_tree_scores, knn_scores)):
    plt.text(i - bar_width / 2, dt_score + 0.03, f"{dt_score:.2f}", ha='center', va='bottom', fontsize=10)
    plt.text(i + bar_width / 2, knn_score + 0.03, f"{knn_score:.2f}", ha='center', va='bottom', fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()
