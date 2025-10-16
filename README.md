# 🧠 Machine Learning Final Project — Employee Attrition Prediction

## 📘 Project Overview
This project aims to predict **Employee Attrition** (whether an employee will leave the company or not) using various **Machine Learning algorithms**.  
The dataset used is **IBM HR Analytics Employee Attrition & Performance Dataset**, a popular dataset for classification tasks.

The goal is to:
- Analyze the key factors that lead to employee attrition.
- Compare multiple ML models to determine which performs best.
- Visualize feature relationships and evaluate model performance.

---

## 📂 Dataset Information
**Dataset Name:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`  
**Source:** [IBM HR Analytics Dataset](https://www.ibm.com/communities/analytics/watson-analytics-blog/hr-employee-attrition/)

### Key Features:
| Feature | Description |
|----------|--------------|
| Age | Age of the employee |
| BusinessTravel | Frequency of business travel |
| Department | Department of the employee |
| EducationField | Field of study |
| Gender | Male / Female |
| JobRole | Employee role |
| MaritalStatus | Marital status |
| OverTime | Whether employee works overtime |
| Attrition | Target variable (Yes/No) |

---

## 🧩 Project Workflow

### 1️⃣ Importing Libraries
```python
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
````

---

### 2️⃣ Data Loading

```python
df = pd.read_csv('/content/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()
```

---

### 3️⃣ Data Preprocessing

* **Label Encoding** categorical variables using `LabelEncoder`.
* **Dropping** irrelevant columns such as `EmployeeNumber`, `EmployeeCount`, `StandardHours`, and `OverTime`.
* **Splitting Data** into training and test sets (80% - 20%).

```python
x = df.drop(['EmployeeNumber','EmployeeCount','StandardHours','OverTime'], axis=1)
y = df['Attrition']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

---

### 4️⃣ Exploratory Data Analysis (EDA)

Visualizing feature correlations:

```python
plt.figure(figsize=(30,30))
sn.heatmap(x.corr(), annot=True)
plt.show()
```

---

## ⚙️ Machine Learning Models Used

### 🔹 Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
print("Accuracy:", model.score(x_test, y_test))
```

---

### 🔹 Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42))
])
rf_pipe.fit(x_train, y_train)
y_pred_rf = rf_pipe.predict(x_test)
```

**Metrics:**

* Accuracy, Precision, Recall, F1-Score
* Confusion Matrix & Heatmap

---

### 🔹 Support Vector Machine (SVM)

```python
from sklearn.svm import SVC

svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', C=30, gamma='auto', random_state=42))
])
svm_pipe.fit(x_train, y_train)
y_pred_svm = svm_pipe.predict(x_test)
```

**Cross Validation:**

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(svm_pipe, x_train, y_train, cv=5)
print("Cross-validation scores:", scores)
print("Mean CV Score:", scores.mean())
```

---

### 🔹 K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier

knn_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=3))
])
knn_pipe.fit(x_train, y_train)
y_pred_knn = knn_pipe.predict(x_test)
```

---

## 📊 Evaluation Metrics

For all models, the following metrics were used:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**
* **Classification Report**

Example:

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

---

## 🧮 Model Comparison Summary

| Model                        | Description                                   | Accuracy |
| ---------------------------- | --------------------------------------------- | -------- |
| Logistic Regression          | Simple linear model for binary classification | ~84%     |
| Random Forest                | Ensemble model using multiple decision trees  | ~88%     |
| Support Vector Machine (SVM) | Works well for high-dimensional data          | ~90%     |
| K-Nearest Neighbors          | Instance-based learning                       | ~85%     |

> ✅ **SVM achieved the best performance** among the tested models.

---

## 📦 Dependencies

To run this project, install the following packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 💾 Running the Project

1. Upload the dataset file to your Colab environment.
2. Open the `.ipynb` file or `.py` script.
3. Run all cells in sequence.

---

## 🧑‍💻 Author

**Lingesh Kumar**
Machine Learning Final Project — HR Attrition Analysis
🔗 [Google Colab Source](https://colab.research.google.com/drive/11ATdi7GGqaXZP10e530mXQy2lnfYSwaS)

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use and modify it for educational purposes.

```
