# Strategic Customer Churn Prediction

## 📈 Business Context
Retaining existing customers is 5-25x cheaper than acquiring new ones. This project builds a Machine Learning pipeline to identify "At-Risk" customers, allowing the business to intervene with retention offers.

## 🛠️ Tech Stack
- **Algorithms:** Random Forest Classifier.
- **Processing:** Scikit-Learn (StandardScaler, LabelEncoder).
- **Data:** Pandas & NumPy.

## 🧠 Key Insights
- **Feature Importance:** The model identified that 'Tenure' and 'Monthly Charges' are the top predictors of churn.
- **Evaluation:** Used F1-Score and Recall to ensure we don't miss customers who are likely to leave.

## 🚀 How to deploy
This script can be wrapped in a **FastAPI** or **Streamlit** dashboard for real-time predictions (Coming in Day 3!).
