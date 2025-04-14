import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load datasets
heart_df = pd.read_csv("heart.csv")
diabetes_df = pd.read_csv("diabetes.csv")

# Feature sets
heart_simple = ['sex', 'exang', 'thal', 'cp', 'thalach','age', 'trestbps']
heart_advanced = ['sex', 'exang', 'thal', 'cp', 'thalach','age', 'trestbps','chol', 'fbs', 'restecg', 'oldpeak', 'slope', 'ca']
heart_target = 'target'

diabetes_simple = ['DiabetesPedigreeFunction', 'Pregnancies', 'BMI','Glucose', 'BloodPressure','Age']
diabetes_advanced = ['DiabetesPedigreeFunction', 'Pregnancies', 'BMI','Glucose', 'BloodPressure','Age','SkinThickness', 'Insulin']
diabetes_target = 'Outcome'

def train_model(X, y, model_type='logistic'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    else:
        model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

# Heart Disease
print("🔴 Heart Disease Prediction")
heart_y = heart_df[heart_target]
acc_simple_heart = train_model(heart_df[heart_simple], heart_y, model_type='logistic')
acc_adv_heart = train_model(heart_df[heart_advanced], heart_y, model_type='logistic')
print(f"Simple Model Accuracy:  {acc_simple_heart * 100:.2f}%")
print(f"Advanced Model Accuracy: {acc_adv_heart * 100:.2f}%\n")

# Diabetes
print("🟣 Diabetes Prediction")
diabetes_y = diabetes_df[diabetes_target]
acc_simple_diabetes = train_model(diabetes_df[diabetes_simple], diabetes_y, model_type='svm')
acc_adv_diabetes = train_model(diabetes_df[diabetes_advanced], diabetes_y, model_type='svm')
print(f"Simple Model Accuracy:  {acc_simple_diabetes * 100:.2f}%")
print(f"Advanced Model Accuracy: {acc_adv_diabetes * 100:.2f}%")
