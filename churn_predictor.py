import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. إنشاء بيانات احترافية (Synthetic Data) تحاكي شركات الاتصالات
np.random.seed(42)
data_size = 1000
data = {
    'tenure': np.random.randint(1, 72, data_size), # عدد أشهر الاشتراك
    'monthly_charges': np.random.uniform(20, 120, data_size), # التكلفة الشهرية
    'total_charges': np.random.uniform(100, 8000, data_size), # إجمالي المبالغ
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], data_size),
    'tech_support': np.random.choice(['Yes', 'No'], data_size),
    'churn': np.random.choice([0, 1], data_size, p=[0.7, 0.3]) # 1 يعني رحل العميل
}

df = pd.DataFrame(data)

# 2. معالجة البيانات (Preprocessing)
# تحويل النصوص إلى أرقام (Encoding)
le = LabelEncoder()
df['contract_type'] = le.fit_transform(df['contract_type'])
df['tech_support'] = le.fit_transform(df['tech_support'])

# 3. تقسيم البيانات
X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. تقيس البيانات (Scaling) - خطوة احترافية
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. تدريب النموذج (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. النتائج
y_pred = model.predict(X_test)
print("--- Professional Churn Model Results ---")
print(classification_report(y_test, y_pred))

# 7. أهمية الميزات (Feature Importance) - لتوضيح كيف يفكر المودل
importances = model.feature_importances_
features = X.columns
for f, s in zip(features, importances):
    print(f"Feature: {f}, Score: {s:.4f}")