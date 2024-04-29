import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

# Load your data
save_dir = r'D:\Macro\ML\figs'
data_path = 'D:\Macro\ML\Data_Subject_B.xlsx'
sheet_name = 'tot2'
df = pd.read_excel(data_path, sheet_name=sheet_name)

# Ensure 'Binary' has integer labels starting from 0
class_labels = df['Binary'].unique()
class_mapping = {label: idx for idx, label in enumerate(class_labels)}
df['Binary'] = df['Binary'].map(class_mapping)

feature_columns = ['Mean1_LAB_L_1',
                   'Mean1_LAB_A_1',
                   'Mean1_LAB_B_1',
                   'Mean1_LAB_L_2',
                   'Mean1_LAB_A_2',
                   'Mean1_LAB_B_2',
                   'Mean1_LAB_L_3',
                   'Mean1_LAB_A_3',
                   'Mean1_LAB_B_3',
                   'Mean1_LAB_L_4',
                   'Mean1_LAB_A_4',
                   'Mean1_LAB_B_4',
                   'Mean1_LAB_L_7',
                   'Mean1_LAB_A_7',
                   'Mean1_LAB_B_7',
                   'Mean1_LAB_L_9',
                   'Mean1_LAB_A_9',
                   'Mean1_LAB_B_9'
                  ]


# Select features from the DataFrame
features = df[feature_columns]
target = df['Binary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

# Initialize the XGBoost model
model = XGBClassifier()

# Define the parameter grid to search
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
}

# Create GridSearchCV object
grid_search = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=10), scoring='accuracy')

# Fit the model with the training data
grid_search.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Use the best model to make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the accuracy of the best model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nBest Parameters: {best_params}")
print(f"Accuracy of the Best Model: {accuracy}")

# Perform k-fold cross-validation
cv_scores = cross_val_score(best_model, features, target, cv=StratifiedKFold(n_splits=10), scoring='accuracy')
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {cv_scores.mean()}")


# Plot feature importance
feature_importance = pd.Series(best_model.feature_importances_, index=feature_columns).sort_values(ascending=False)
top_feature_importance = feature_importance[:5]
plt.figure(figsize=(6, 5))
ax = sns.barplot(x=top_feature_importance, y=top_feature_importance.index, palette='viridis')
ax.set_yticklabels([])
for i, p in enumerate(ax.patches):
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(top_feature_importance.index[i], (x + width/2, y + height/2), ha='center', va='center', fontsize=20, fontname='Arial')
#plt.title('Feature Importance Human Subject B (Top 5)', fontsize=20, fontname='Arial')
plt.xlabel('Importance Score', fontsize=20, fontname='Arial')
ax.tick_params(axis='x', labelsize=20)
plt.tight_layout()
file_name = f"Feature_Importance_Plot_{sheet_name}.png"
save_path = os.path.join(save_dir, file_name)
plt.savefig(save_path)
plt.close()

# Print features in order starting from most important
#sorted_feature_importance = sorted(zip(feature_columns, best_model.feature_importances_), key=lambda x: x[1], reverse=True)
#for feature, importance in sorted_feature_importance:
#    print(f"{feature}: {importance}")

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
annot_kws = {"fontsize": 20}  # Set the font size to 20
heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', annot_kws=annot_kws)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=20, fontname='Arial')
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=20, fontname='Arial')
#plt.title('Confusion Matrix Human Subject B (all features)', fontsize=20, fontname='Arial')
plt.xlabel('Predicted', fontsize=20, fontname='Arial')  # Adjust font size as needed
plt.ylabel('True', fontsize=20, fontname='Arial')  # Adjust font size as needed
plt.tight_layout()
file_name = f"Confusion_Matrix_{sheet_name}.png"
save_path = os.path.join(save_dir, file_name)
plt.savefig(save_path)
plt.close()

# Plot learning curves
train_sizes, train_scores, test_scores = learning_curve(best_model, X_train, y_train, cv=StratifiedKFold(n_splits=10), train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')
plt.figure(figsize=(6, 5))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Accuracy')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation Accuracy')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title('Learning Curves Bacteria Human Subject B (all features)', fontsize=20, fontname='Arial')
plt.xlabel('Training Examples', fontsize=20, fontname='Arial')  # Adjust font size as needed
plt.ylabel('Accuracy', fontsize=20, fontname='Arial')  # Adjust font size as needed
plt.legend(prop={'size': 20, 'family': 'Arial'})
plt.tight_layout()
file_name = f"Learning_Curve_{sheet_name}.png"
save_path = os.path.join(save_dir, file_name)
plt.savefig(save_path)
plt.close()


# Evaluation Metrics
print("Evaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
