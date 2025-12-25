# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score)

# Display settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Load data
df = pd.read_csv('diabetes.csv')
print(" Data Information:")
print(f"Data size: {df.shape}")
print(f"\nClass distribution:\n{df['Outcome'].value_counts()}")
print(f"\nPositive class percentage: {(df['Outcome'].mean()*100):.1f}%")

# 2. Simple preprocessing
# Replace illogical zero values
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    df[col] = df[col].replace(0, df[col].median())

# 3. Split data
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train model with class_weight to handle imbalance
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',  # Important for imbalanced data
    solver='liblinear'
)
model.fit(X_train_scaled, y_train)

# 6. Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Class 1 probabilities

#  Comprehensive Evaluation Section

print("\n" + "="*50)
print(" Comprehensive Logistic Regression Model Evaluation")
print("="*50)

# A) Main metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n Main Metrics:")
print(f" Accuracy: {accuracy:.3f}")
print(f" Precision: {precision:.3f}")
print(f" Recall: {recall:.3f}")
print(f" F1-Score: {f1:.3f}")

# B) Complete classification report
print(f"\n Complete Classification Report:")
print(classification_report(y_test, y_pred, 
                           target_names=['No Diabetes', 'Has Diabetes'],
                           digits=3))

# C) Detailed confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n Confusion Matrix Details:")
print(f"True Positive (TP): {tp}")
print(f"True Negative (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")

# Calculate additional metrics from confusion matrix
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

print(f"\n Additional Metrics:")
print(f"Specificity: {specificity:.3f}")
print(f"Negative Predictive Value (NPV): {npv:.3f}")


#  Visualization Section

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Confusion Matrix Heatmap
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
ax1.set_ylabel('Actual', fontsize=12)
ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax1.set_xticklabels(['No Diabetes', 'Has Diabetes'])
ax1.set_yticklabels(['No Diabetes', 'Has Diabetes'])

# 2. Metrics Comparison Bar Chart
ax2 = axes[0, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']

bars = ax2.bar(metrics, values, color=colors, edgecolor='black')
ax2.set_title('Evaluation Metrics Comparison', fontsize=12, fontweight='bold')
ax2.set_ylabel('Value', fontsize=12)
ax2.set_ylim(0, 1)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{value:.3f}', ha='center', va='bottom', fontsize=11)

# 3. ROC Curve
ax3 = axes[1, 0]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

ax3.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax3.set_xlabel('False Positive Rate', fontsize=12)
ax3.set_ylabel('True Positive Rate', fontsize=12)
ax3.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax3.legend(loc="lower right")
ax3.grid(True, alpha=0.3)

# 4. Precision-Recall Curve
ax4 = axes[1, 1]
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

ax4.plot(recall_vals, precision_vals, color='purple', lw=2,
         label=f'Precision-Recall (AP = {avg_precision:.3f})')
ax4.set_xlabel('Recall', fontsize=12)
ax4.set_ylabel('Precision', fontsize=12)
ax4.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
ax4.legend(loc="best")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


#  Model Interpretation
print("\n" + "="*50)
print(" Model Interpretation")
print("="*50)

# Model coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0],
    'Abs_Coefficient': np.abs(model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\n Feature Importance (by absolute coefficient values):")
print(coefficients.drop('Abs_Coefficient', axis=1).to_string(index=False))

# Coefficient plot
plt.figure(figsize=(10, 6))
colors = ['red' if coef < 0 else 'green' for coef in coefficients['Coefficient']]
plt.barh(coefficients['Feature'], coefficients['Coefficient'], color=colors)
plt.xlabel('Coefficient Value', fontsize=12)
plt.title('Logistic Regression Coefficients\n(Negative: decreases risk, Positive: increases risk)', 
          fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# Threshold Analysis

print(" Threshold Analysis")
print("="*50)

# Performance at different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print("\nThreshold | Precision | Recall | F1-Score")
print("-" * 40)

for thresh in thresholds:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    p = precision_score(y_test, y_pred_thresh, zero_division=0)
    r = recall_score(y_test, y_pred_thresh)
    f = f1_score(y_test, y_pred_thresh)
    print(f"{thresh:.1f}       | {p:.3f}     | {r:.3f}  | {f:.3f}")

#  Summary
print("\n" + "="*50)
print(" Results Summary")
print("="*50)

print(f"\n Model Strengths:")
if recall > 0.7:
    print(f"  • High Recall ({recall:.3f}): Good ability to identify actual patients")
if precision > 0.7:
    print(f"  • High Precision ({precision:.3f}): Good accuracy in positive predictions")
if f1 > 0.7:
    print(f"  • Good balance between Precision and Recall (F1-Score: {f1:.3f})")

print(f"\n  Areas for Improvement:")
if fp > fn:
    print(f"  • {fp} False Positives: May misclassify healthy people as diabetic")
if fn > fp:
    print(f"  • {fn} False Negatives: May miss actual diabetic patients")

print(f"\n Recommendation:")
if recall < precision:
    print("  • If identifying all patients is important, lower the threshold")
else:
    print("  • If avoiding false alarms is important, increase the threshold")


#  Final Model Performance Summary Table

print("\n" + "="*50)
print(" Final Performance Summary")
print("="*50)

summary_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 
               'Specificity', 'NPV', 'ROC-AUC'],
    'Value': [accuracy, precision, recall, f1, 
              specificity, npv, roc_auc],
    'Interpretation': [
        'Overall correctness',
        'Correct positive predictions',
        'Ability to find all positives',
        'Balance of Precision and Recall',
        'Ability to identify negatives',
        'Correct negative predictions',
        'Overall classification ability'
    ]
})

print(summary_df.to_string(index=False))
