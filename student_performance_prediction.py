
# Student Performance Prediction - Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample Dataset: Study Hours, Attendance and Final Score
data = {
    'Study_Hours': [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 8, 9, 10],
    'Attendance': [60, 65, 65, 70, 75, 80, 82, 85, 90, 92, 94, 96, 98, 100],
    'Score': [45, 48, 50, 55, 57, 60, 65, 66, 70, 75, 78, 85, 88, 90]
}
df = pd.DataFrame(data)

# Feature and Target
X = df[['Study_Hours', 'Attendance']]
y = df['Score']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))

# Plotting
plt.scatter(df['Study_Hours'], df['Score'], color='blue', label='Actual')
plt.plot(df['Study_Hours'], model.predict(df[['Study_Hours', 'Attendance']]), color='red', label='Predicted')
plt.xlabel('Study Hours')
plt.ylabel('Score')
plt.title('Study Hours vs Score')
plt.legend()
plt.show()
