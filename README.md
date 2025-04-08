
# Employee Attrition Prediction

This project uses machine learning to predict employee attrition based on various HR-related features. The dataset used is the IBM HR Analytics dataset available on Kaggle.

## 📊 Dataset

The dataset includes features such as:
- Age
- Gender
- Department
- Job Role
- OverTime
- Monthly Income
- Job Satisfaction
- Work Life Balance
- Years at Company
- Distance from Home
- And more...

The target variable is **Attrition** (Yes/No), which indicates whether an employee left the company.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Random Forest Classifier

## 🔍 Workflow

1. Data Loading
2. Preprocessing & Encoding
3. Feature Scaling
4. Model Training (Random Forest)
5. Evaluation using Accuracy, Classification Report, and Confusion Matrix
6. Feature Importance Analysis

## 📈 Results

- Accuracy: ~85-90% depending on model and tuning
- Key factors influencing attrition include:
  - Overtime
  - Job Satisfaction
  - Monthly Income
  - Years at Company

## 📂 How to Run

1. Clone the repository
2. Make sure you have the required libraries installed
3. Place the dataset as `HR_dataset.csv` in the project directory
4. Run `employee_attrition_prediction.py`

```bash
python employee_attrition_prediction.py
```

## 📎 Dataset Source

[Kaggle: IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

## 📧 Contact

For questions or suggestions, feel free to reach out!
