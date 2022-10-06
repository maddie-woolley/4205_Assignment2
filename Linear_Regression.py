# part 1-This is done
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

diabetes = pd.read_csv('diabetes.csv')
# imports data set and prints the main data, as well as the description of dataset


# print(diabetes)
# print(diabetes.describe())

# part 2
corrMatrix = diabetes.corr()
# print(corrMatrix)

# sb.heatmap(corrMatrix, annot=True)
# plt.show()

# part 3A - countplot
# sb.countplot(x='Outcome', hue='Outcome', data=diabetes)
# plt.legend(labels = ['Negative', 'Positive'])
# plt.tight_layout()
# plt.show()

# part 3B - distribution plot
# sb.displot(diabetes, kind="kde")
# plt.show()

# part 3C - box plot
# sb.boxplot(diabetes)
# plt.show()

# part 3D - Pairplot
# sb.pairplot(diabetes)
# plt.show()

# part 3E - Violin Plot
# sb.violinplot(diabetes)
# plt.show()

# Part 4 -
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

x = diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = diabetes['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = LogisticRegression()
model.fit(x_train, y_train)
print(model.score(x_train, y_train))
confMatrix = confusion_matrix(y_train, model.predict(x_train))
print(confMatrix)
# sb.heatmap(confMatrix / np.sum(confMatrix), annot=True, fmt='.2%', )
# plt.show()

# Part 4 AUC /ROC Curve

# define metrics
y_pred_proba = model.predict_proba(x_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

# create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()
