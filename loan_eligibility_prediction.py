
# Loan Eligibility Prediction - Decision Tree Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Sample Data
data = {
    'Income': [25000, 40000, 30000, 50000, 60000, 35000, 70000, 20000, 45000, 48000],
    'Credit_Score': [300, 650, 600, 700, 720, 610, 750, 400, 680, 700],
    'Education': ['Graduate', 'Not Graduate', 'Graduate', 'Graduate', 'Not Graduate', 'Graduate', 'Graduate', 'Not Graduate', 'Graduate', 'Not Graduate'],
    'Loan_Status': ['N', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y', 'Y']
}
df = pd.DataFrame(data)

# Encoding
le = LabelEncoder()
df['Education'] = le.fit_transform(df['Education'])
df['Loan_Status'] = df['Loan_Status'].map({'Y':1, 'N':0})

# Features and Target
X = df[['Income', 'Credit_Score', 'Education']]
y = df['Loan_Status']

# Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
