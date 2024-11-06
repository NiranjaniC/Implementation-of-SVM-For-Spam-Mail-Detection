# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and Preprocess Data: Load the dataset, select relevant columns, and map labels to numeric values.

2.Split Data: Divide the dataset into training and test sets.

3.Vectorize Text: Convert text messages into a numerical format using TfidfVectorizer.

4.Train SVM Model: Fit an SVM model with a linear kernel on the training data.

5.Evaluate Model: Predict on the test set and print the accuracy score and classification report.

6.Visualize Results: Plot the confusion matrix, ROC curve, and precision-recall curve for detailed evaluation.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Niranjani.C
RegisterNumber:  212223220069
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]  # Select relevant columns
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # Map labels to 0 (ham) and 1 (spam)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)
y_proba = model.decision_function(X_test_vec)  # For ROC and Precision-Recall curves

# 1. Accuracy Score
print("Accuracy:", accuracy_score(y_test, y_pred))

# 2. Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 4. ROC Curve and AUC Score
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 5. Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()
```

## Output:
![Screenshot 2024-11-06 093420](https://github.com/user-attachments/assets/79efa94e-f891-457a-82d0-10adac89735c)
![Screenshot 2024-11-06 093439](https://github.com/user-attachments/assets/0e53b13f-6ae5-4e1f-95ab-217117b72caa)
![Screenshot 2024-11-06 093453](https://github.com/user-attachments/assets/d11c8db5-5a07-461e-a23c-e874043b7cad)
![Screenshot 2024-11-06 093504](https://github.com/user-attachments/assets/ad93b019-861e-4df7-b69d-4babdf26d096)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
