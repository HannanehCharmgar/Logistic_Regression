# پروژه پیش بینی دیابت
# بارگذاری کتابخانه ها
در این سلول تمام کتابخانه‌های مورد نیاز پروژه بارگذاری می‌شوند.  
این کتابخانه‌ها شامل ابزارهای کار با داده‌ها (`pandas`, `numpy`)، مصورسازی (`matplotlib`, `seaborn`) و مدل‌سازی و ارزیابی یادگیری ماشین (`scikit-learn`) هستند.  
همچنین تنظیمات ظاهری نمودارها انجام می‌شود تا خروجی‌ها خواناتر و حرفه‌ای‌تر باشند.
```
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

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
```
## بارگذاری و بررسی داده
در این بخش داده‌ها از فایل CSV بارگذاری شده و یک بررسی اولیه انجام می‌شود.  
هدف این مرحله، درک ساختار داده‌ها، تعداد نمونه‌ها و توزیع کلاس‌هاست.  
بررسی درصد کلاس مثبت (بیماران دیابتی) برای تشخیص عدم توازن داده‌ها اهمیت دارد.
```
import pandas as pd
df = pd.read_csv('diabetes.csv')

print(" Data Information:")
print(f"Data size: {df.shape}")
print(f"\nClass distribution:\n{df['Outcome'].value_counts()}")
print(f"\nPositive class percentage: {(df['Outcome'].mean()*100):.1f}%")
```
## پاکسازی داده
در این سلول پیش‌پردازش اولیه داده‌ها انجام می‌شود.  
برخی ویژگی‌های پزشکی دارای مقدار صفر هستند که از نظر علمی معتبر نیستند.  
این مقادیر صفر با میانه هر ستون جایگزین می‌شوند تا کیفیت داده‌ها افزایش یابد.
```
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    df[col] = df[col].replace(0, df[col].median())
```
## تفکیک داده به آموزش_تست و استاندارسازی
در این مرحله داده‌ها به ویژگی‌ها (`X`) و برچسب‌ها (`y`) تقسیم می‌شوند.  
سپس داده‌ها به مجموعه آموزش و تست تفکیک می‌شوند.  
استفاده از `stratify=y` باعث حفظ نسبت کلاس‌ها در هر دو مجموعه می‌شود.
در این سلول ویژگی‌ها استانداردسازی می‌شوند تا میانگین صفر و واریانس یک داشته باشند.  
استانداردسازی برای مدل‌هایی مانند Logistic Regression ضروری است و از تأثیر نامتعادل ویژگی‌ها با مقیاس بزرگ جلوگیری می‌کند.
```
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
## آموزش مدل
در این بخش مدل رگرسیون لجستیک آموزش داده می‌شود.  
استفاده از `class_weight='balanced'` باعث توجه بیشتر مدل به کلاس اقلیت می‌شود.  
پارامتر `max_iter` برای اطمینان از همگرایی مدل تنظیم شده است.
```
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',
    solver='liblinear'
)
model.fit(X_train_scaled, y_train)
```
## پیش بینی خروجی مدل
در این سلول خروجی مدل تولید می‌شود.  
هم پیش‌بینی نهایی کلاس‌ها و هم احتمال تعلق هر نمونه به کلاس مثبت محاسبه می‌شود.  
احتمال‌ها در مراحل بعدی برای تحلیل دقیق‌تر استفاده می‌شوند.
```
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
```

## ارزیابی عملکرد مدل
در این بخش عملکرد مدل با معیارهای مختلف ارزیابی می‌شود.  
معیارهایی مانند Accuracy، Precision، Recall و F1-Score محاسبه می‌شوند.  
همچنین با استفاده از ماتریس درهم‌ریختگی، معیارهای پزشکی مهم استخراج می‌شوند.
```
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n Main Metrics:")
print(f" Accuracy: {accuracy:.3f}")
print(f" Precision: {precision:.3f}")
print(f" Recall: {recall:.3f}")
print(f" F1-Score: {f1:.3f}")

print(f"\n Complete Classification Report:")
print(classification_report(y_test, y_pred, 
                           target_names=['No Diabetes', 'Has Diabetes'],
                           digits=3))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n Confusion Matrix Details:")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\n Additional Metrics:")
print(f"Specificity: {specificity:.3f}")
print(f"NPV: {npv:.3f}")
```
## مصورسازی نتایج و ارزیابی ها
در این سلول نتایج مدل به صورت بصری نمایش داده می‌شوند.  
نمودارها شامل ماتریس درهم‌ریختگی، مقایسه معیارها، منحنی ROC و Precision-Recall هستند.  
این نمایش‌ها درک شهودی بهتری از عملکرد مدل فراهم می‌کنند.
```
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0,0])
axes[0,0].set_ylabel('Actual')
axes[0,0].set_title('Confusion Matrix')
axes[0,0].set_xticklabels(['No Diabetes', 'Has Diabetes'])
axes[0,0].set_yticklabels(['No Diabetes', 'Has Diabetes'])

# Metrics Bar Chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
bars = axes[0,1].bar(metrics, values, color=colors, edgecolor='black')
axes[0,1].set_ylim(0, 1)
axes[0,1].set_title('Evaluation Metrics Comparison')
for bar, value in zip(bars, values):
    axes[0,1].text(bar.get_x() + bar.get_width()/2., value + 0.02,
                   f'{value:.3f}', ha='center')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
axes[1,0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[1,0].plot([0,1],[0,1], color='navy', lw=2, linestyle='--', label='Random')
axes[1,0].set_xlabel('False Positive Rate')
axes[1,0].set_ylabel('True Positive Rate')
axes[1,0].set_title('ROC Curve')
axes[1,0].legend()

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)
axes[1,1].plot(recall_vals, precision_vals, color='purple', lw=2,
               label=f'Precision-Recall (AP = {avg_precision:.3f})')
axes[1,1].set_xlabel('Recall')
axes[1,1].set_ylabel('Precision')
axes[1,1].set_title('Precision-Recall Curve')
axes[1,1].legend()

plt.tight_layout()
plt.show()
```
## تحلیل مدل
در این بخش مدل از نظر تأثیر ویژگی‌ها تفسیر می‌شود.  
ضرایب Logistic Regression نشان می‌دهند هر ویژگی چگونه بر ریسک دیابت اثر می‌گذارد.  
این مرحله برای توضیح‌پذیری و تحلیل پزشکی بسیار مهم است.
```
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0],
    'Abs_Coefficient': np.abs(model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\n Feature Importance:")
print(coefficients.drop('Abs_Coefficient', axis=1).to_string(index=False))

plt.figure(figsize=(10,6))
colors = ['red' if coef < 0 else 'green' for coef in coefficients['Coefficient']]
plt.barh(coefficients['Feature'], coefficients['Coefficient'], color=colors)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression Coefficients\n(Negative: decreases risk, Positive: increases risk)')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
```
## تحلیل آستانه تصمیم گیری ( threshold)
در این سلول تأثیر تغییر آستانه تصمیم‌گیری بررسی می‌شود.  
با تغییر threshold می‌توان تعادل بین Precision و Recall را کنترل کرد.  
این تحلیل در مسائل پزشکی که هزینه خطاها متفاوت است اهمیت زیادی دارد.
```
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print("\nThreshold | Precision | Recall | F1-Score")
print("-" * 40)
for thresh in thresholds:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    p = precision_score(y_test, y_pred_thresh, zero_division=0)
    r = recall_score(y_test, y_pred_thresh)
    f = f1_score(y_test, y_pred_thresh)
    print(f"{thresh:.1f}       | {p:.3f}     | {r:.3f}  | {f:.3f}")
```
## تحلیل نهایی
در این بخش یک جمع‌بندی تحلیلی از عملکرد مدل ارائه می‌شود.  
نقاط قوت، نقاط ضعف و نوع خطاهای غالب مدل بررسی می‌شوند.  
در نهایت، توصیه‌ای عملی برای بهبود تصمیم‌گیری مدل ارائه می‌شود.
```
print("\n Model Strengths and Weaknesses:")
if recall > 0.7:
    print(f"  • High Recall ({recall:.3f})")
if precision > 0.7:
    print(f"  • High Precision ({precision:.3f})")
if f1 > 0.7:
    print(f"  • Balanced F1-Score ({f1:.3f})")

if fp > fn:
    print(f"  • {fp} False Positives")
if fn > fp:
    print(f"  • {fn} False Negatives")

print("\n Recommendation:")
if recall < precision:
    print("  • Lower the threshold to identify more patients")
else:
    print("  • Increase threshold to reduce false alarms")
```
## جدول عملکرد نهایی مدل
در این بخش تمام معیارهای نهایی مدل در قالب یک جدول خلاصه می‌شوند.  
این جدول یک نمای کلی و سریع از عملکرد مدل ارائه می‌دهد  
و برای گزارش‌دهی و مرور سریع بسیار کاربردی است.
```
summary_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 
               'Specificity', 'NPV', 'ROC-AUC'],
    'Value': [accuracy, precision, recall, f1, specificity, npv, roc_auc],
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
```
